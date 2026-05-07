import argparse
import importlib
import json
import os
import re
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent

MODULES_TO_PURGE = [
    "app",
    "config",
    "diagnostic_logic",
    "engine",
    "prompts",
    "syllabus",
    "tutor",
    "rag",
    "retriever",
    "learner_store",
]

LEVEL_RANK = {"Pre-A1": 0, "A1": 1, "A2": 2, "B1": 3}
LEXI_INVOKE_TIMEOUT_SECONDS = int(os.getenv("LEXI_INVOKE_TIMEOUT", "120"))

DIAGNOSIS_SCENARIOS = [
    {
        "name": "diagnose_a1_foundation",
        "gold_level": "A1",
        "learner_profile": {
            "true_level": "A1",
            "strengths": ["negation_kein"],
            "weaknesses": ["perfect_tense_basics", "subordinate_clause_weil"],
            "style": "short",
        },
    },
    {
        "name": "diagnose_a2_everyday",
        "gold_level": "A2",
        "learner_profile": {
            "true_level": "A2",
            "strengths": ["perfect_tense_basics", "comparatives_basics"],
            "weaknesses": ["konjunktiv_ii_basics"],
            "style": "short",
        },
    },
    {
        "name": "diagnose_b1_structured",
        "gold_level": "B1",
        "learner_profile": {
            "true_level": "B1",
            "strengths": ["subordinate_clause_weil", "relative_clauses_basics"],
            "weaknesses": ["konjunktiv_ii_basics"],
            "style": "short",
        },
    },
]

TUTORING_SCENARIOS = [
    {
        "name": "a1_articles_help",
        "level": "A1",
        "messages": [
            "Please explain the difference between ein, eine, and einen very simply.",
            "Give me one easy practice sentence.",
        ],
    },
    {
        "name": "a2_perfekt_practice",
        "level": "A2",
        "messages": [
            "I find Perfekt difficult. Please explain it with one short example.",
            "Now give me a short exercise.",
        ],
    },
    {
        "name": "b1_work_communication",
        "level": "B1",
        "messages": [
            "Help me improve German for work communication.",
            "Please explain how to use weil clauses in a more natural way.",
        ],
    },
]

CONTINUITY_SCENARIOS = [
    {
        "name": "a2_returning_perfekt",
        "level": "A2",
        "learner_id": "benchmark_a2_returning",
        "display_name": "Benchmark A2",
        "session1_messages": [
            "I want to practice Perfekt for daily life.",
            "Please give me one short exercise about yesterday.",
        ],
        "session2_messages": [
            "Can we continue from where we stopped yesterday?",
            "Give me another short exercise now.",
        ],
        "memory_keywords": ["perfekt", "yesterday", "exercise"],
    },
    {
        "name": "b1_returning_work_focus",
        "level": "B1",
        "learner_id": "benchmark_b1_returning",
        "display_name": "Benchmark B1",
        "session1_messages": [
            "I want to improve German for work communication.",
            "Please explain weil clauses and then give me one practice question.",
        ],
        "session2_messages": [
            "Can we continue from where we stopped yesterday?",
            "Please give me a short work-related exercise.",
        ],
        "memory_keywords": ["work", "weil", "practice"],
    },
]


@dataclass
class BenchmarkResult:
    system_name: str
    suite: str
    scenario_name: str
    success: bool
    error: str
    gold_level: str
    detected_level: str
    tutor_level: str
    diagnostic_trace: List[Dict[str, Any]]
    session1_user: List[str]
    session1_assistant: List[str]
    session2_user: List[str]
    session2_assistant: List[str]
    retrieval_stats: Dict[str, Any]
    metrics: Dict[str, Any]


@contextmanager
def project_context(project_path: Path):
    prev_cwd = Path.cwd()
    prev_sys_path = list(sys.path)
    os.chdir(project_path)
    extra_paths = [str(project_path)]

    venv_lib = project_path / "venv" / "lib"
    if venv_lib.exists():
        for candidate in sorted(venv_lib.glob("python*/site-packages")):
            extra_paths.append(str(candidate))

    for path in reversed(extra_paths):
        sys.path.insert(0, path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        sys.path = prev_sys_path


def purge_modules() -> None:
    for name in MODULES_TO_PURGE:
        if name in sys.modules:
            del sys.modules[name]


def _empty_retrieval_stats() -> Dict[str, Any]:
    return {
        "turns": 0,
        "turns_with_chunks": 0,
        "total_chunks": 0,
        "fallback_turns": 0,
    }


def _normalize_level(level: str) -> str:
    if level in {"A1", "A2", "B1"}:
        return level
    if level == "Pre-A1":
        return "A1"
    return "A1"


def _clamp_level(level: str) -> str:
    return level if level in LEVEL_RANK else "A1"


def _rank(level: str) -> int:
    return LEVEL_RANK.get(_clamp_level(level), LEVEL_RANK["A1"])


def _extract_visible_question(text: str) -> str:
    match = re.search(
        r"Level check for\s+([A-Za-z0-9\-]+)\s+\((.*?)\s+-\s+(.*?)\):\s*(.*)",
        text,
        flags=re.DOTALL,
    )
    if match:
        return match.group(4).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _choose_quality(learner_profile: Dict[str, Any], target_level: str, grammar_point: str) -> str:
    true_level = _clamp_level(learner_profile.get("true_level", "A1"))
    strengths = set(learner_profile.get("strengths", []))
    weaknesses = set(learner_profile.get("weaknesses", []))

    quality_score = _rank(true_level) - _rank(target_level)
    if grammar_point in strengths:
        quality_score += 1
    if grammar_point in weaknesses:
        quality_score -= 1

    if quality_score >= 0:
        return "full"
    if quality_score == -1:
        return "partial"
    return "fail"


def _template_answer(grammar_point: str, quality: str, visible_question: str) -> str:
    bank = {
        "basic_greeting_phrase": {"full": "Hallo, ich heiße Anna.", "partial": "Hallo ich Anna.", "fail": "Hallo."},
        "yes_no_basic_response": {"full": "Ja, ein bisschen.", "partial": "Ja bisschen.", "fail": "Ja."},
        "single_word_everyday_vocab": {"full": "Wasser.", "partial": "Wasser trinken.", "fail": "Essen."},
        "indefinite_articles_ein_eine_einen": {
            "full": "Der Mann isst einen Apfel.",
            "partial": "Der Mann isst ein Apfel.",
            "fail": "Der Mann isst Apfel.",
        },
        "negation_kein": {
            "full": "Ich habe kein Auto.",
            "partial": "Ich habe nicht Auto.",
            "fail": "Ich habe Auto.",
        },
        "present_tense_basic_verbs": {
            "full": "Ich wohne in Berlin.",
            "partial": "Ich wohnen in Berlin.",
            "fail": "Berlin.",
        },
        "perfect_tense_basics": {
            "full": "Ich habe gestern Deutsch gelernt.",
            "partial": "Ich lerne gestern Deutsch.",
            "fail": "Ich lerne Deutsch.",
        },
        "accusative_with_movement": {
            "full": "Ich lege das Buch auf den Tisch.",
            "partial": "Ich lege das Buch auf dem Tisch.",
            "fail": "Das Buch ist auf dem Tisch.",
        },
        "comparatives_basics": {
            "full": "Ein Auto ist schneller als ein Fahrrad.",
            "partial": "Ein Auto ist mehr schnell als ein Fahrrad.",
            "fail": "Auto und Fahrrad sind gut.",
        },
        "subordinate_clause_weil": {
            "full": "Ich lerne Deutsch, weil ich in Deutschland arbeiten will.",
            "partial": "Ich lerne Deutsch, weil ich will in Deutschland arbeiten.",
            "fail": "Ich lerne Deutsch. Ich will in Deutschland arbeiten.",
        },
        "konjunktiv_ii_basics": {
            "full": "Ich würde mehr reisen, wenn ich Zeit hätte.",
            "partial": "Ich werde mehr reisen, wenn ich Zeit hätte.",
            "fail": "Ich reise mehr.",
        },
        "relative_clauses_basics": {
            "full": "Das ist die Frau, die mir hilft.",
            "partial": "Das ist die Frau die hilft mir.",
            "fail": "Das ist eine Frau. Sie hilft mir.",
        },
    }

    if grammar_point in bank:
        return bank[grammar_point][quality]

    lowered = visible_question.lower()
    if "weil" in lowered:
        return bank["subordinate_clause_weil"][quality]
    if "wuerde" in lowered or "würde" in lowered or "hypothetical" in lowered:
        return bank["konjunktiv_ii_basics"][quality]
    if "relative" in lowered:
        return bank["relative_clauses_basics"][quality]
    if "perfekt" in lowered or "yesterday" in lowered:
        return bank["perfect_tense_basics"][quality]
    if "kein" in lowered:
        return bank["negation_kein"][quality]
    if "als" in lowered or "compare" in lowered:
        return bank["comparatives_basics"][quality]
    return {"full": "Ich lerne Deutsch jeden Tag.", "partial": "Ich lernen Deutsch jeden Tag.", "fail": "Deutsch."}[quality]


def simulate_learner_answer(question_text: str, learner_profile: Dict[str, Any], target_level: str, grammar_point: str) -> str:
    quality = _choose_quality(learner_profile, target_level, grammar_point)
    return _template_answer(grammar_point, quality, _extract_visible_question(question_text))


def _metrics_from_sessions(session1: List[str], session2: List[str], memory_keywords: List[str]) -> Dict[str, Any]:
    all_replies = session1 + session2
    if not all_replies:
        return {
            "assistant_turns": 0,
            "avg_response_chars": 0,
            "followup_question_rate": 0.0,
            "session2_memory_keyword_hits": 0,
            "session2_memory_recall": False,
        }

    avg_len = sum(len(item) for item in all_replies) / len(all_replies)
    followups = sum(1 for item in all_replies if "?" in item)
    session2_blob = "\n".join(session2).lower()
    hits = sum(1 for kw in memory_keywords if kw.lower() in session2_blob)
    return {
        "assistant_turns": len(all_replies),
        "avg_response_chars": round(avg_len, 1),
        "followup_question_rate": round(followups / len(all_replies), 3),
        "session2_memory_keyword_hits": hits,
        "session2_memory_recall": hits > 0,
    }


def _minimal_summary(level: str) -> Dict[str, Any]:
    return {
        "detected_level": level,
        "strengths": [f"{level} learner profile available"],
        "weaknesses": [],
    }


def _is_transient_error(message: str) -> bool:
    lowered = (message or "").lower()
    return any(token in lowered for token in ["502", "503", "504", "timeout", "bad gateway", "gateway timeout"])


def _with_retries(runner: Callable[..., BenchmarkResult], *args: Any, attempts: int = 3, **kwargs: Any) -> BenchmarkResult:
    last_result: Optional[BenchmarkResult] = None
    for idx in range(attempts):
        result = runner(*args, **kwargs)
        if result.success:
            return result
        last_result = result
        if idx >= attempts - 1 or not _is_transient_error(result.error):
            return result
        time.sleep(float(idx + 1))
    return last_result if last_result is not None else runner(*args, **kwargs)


@contextmanager
def _time_limit(seconds: int):
    if seconds <= 0:
        yield
        return

    def _handle_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Timed out after {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _lexi_invoke_with_timeout(app_obj: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    with _time_limit(LEXI_INVOKE_TIMEOUT_SECONDS):
        return app_obj.invoke(state)


def run_diagnosis_zeroshot(scenario: Dict[str, Any]) -> BenchmarkResult:
    system = "Zeroshot"
    try:
        with project_context(ROOT / "Zeroshot"):
            purge_modules()
            diagnostic_logic = importlib.import_module("diagnostic_logic")
            results_by_task: Dict[int, int] = {}
            trace: List[Dict[str, Any]] = []
            current_id = diagnostic_logic.DiagnosticManager.get_start_task_id()
            current_level = diagnostic_logic.DiagnosticManager.get_level_for_task(current_id) if current_id is not None else "A1"

            while current_id is not None:
                task = diagnostic_logic.DiagnosticManager.get_task(current_id)
                if not task:
                    break
                question = diagnostic_logic.generate_diagnostic_question(task, user_level=current_level)
                answer = simulate_learner_answer(question, scenario["learner_profile"], task["level"], task["grammar_point"])
                evaluation = diagnostic_logic.grade_diagnostic_answer(task, answer)
                results_by_task[current_id] = evaluation["score_value"]
                trace.append(
                    {
                        "task_id": task["id"],
                        "target_level": task["level"],
                        "topic": task["topic"],
                        "grammar_point": task["grammar_point"],
                        "question": question,
                        "learner_answer": answer,
                        "score_value": evaluation["score_value"],
                    }
                )
                next_id = diagnostic_logic.DiagnosticManager.get_next_task_id(current_id, results_by_task)
                if next_id is not None:
                    current_level = diagnostic_logic.DiagnosticManager.get_level_for_task(next_id)
                current_id = next_id

            detected = _normalize_level(diagnostic_logic.DiagnosticManager.determine_final_level(results_by_task))
            return BenchmarkResult(system, "diagnosis", scenario["name"], True, "", scenario["gold_level"], detected, "", trace, [], [], [], [], _empty_retrieval_stats(), {})
    except Exception as exc:
        return BenchmarkResult(system, "diagnosis", scenario["name"], False, str(exc), scenario["gold_level"], "", "", [], [], [], [], [], _empty_retrieval_stats(), {})


def run_diagnosis_llmrag(scenario: Dict[str, Any]) -> BenchmarkResult:
    system = "llmrag"
    try:
        with project_context(ROOT / "llmrag"):
            purge_modules()
            diagnostic_logic = importlib.import_module("diagnostic_logic")
            attempts: List[Dict[str, Any]] = []
            trace: List[Dict[str, Any]] = []
            decision: Dict[str, str] = {"action": "continue", "final_level": "A1"}

            while len(attempts) < getattr(diagnostic_logic, "MAX_ATTEMPTS", 6):
                plan = diagnostic_logic._plan_next_question(attempts)
                question = diagnostic_logic._generate_question(plan, attempts)
                answer = simulate_learner_answer(question, scenario["learner_profile"], plan["target_level"], plan["grammar_point"])
                grade = diagnostic_logic._grade_answer(plan, question, answer, attempts)
                attempt = {
                    "attempt": len(attempts) + 1,
                    "question_level": plan["target_level"],
                    "focus_topic": plan["focus_topic"],
                    "grammar_point": plan["grammar_point"],
                    "goal": plan["goal"],
                    "question": question,
                    "answer": answer,
                    "label": grade["label"],
                    "score": grade["score"],
                    "estimated_level": grade["estimated_level"],
                    "rationale": grade["rationale"],
                }
                attempts.append(attempt)
                trace.append(
                    {
                        "attempt": attempt["attempt"],
                        "target_level": attempt["question_level"],
                        "topic": attempt["focus_topic"],
                        "grammar_point": attempt["grammar_point"],
                        "question": question,
                        "learner_answer": answer,
                        "score_value": grade["score"],
                    }
                )
                decision = diagnostic_logic._decide_next_step(attempts)
                if decision["action"] == "stop":
                    break

            detected = _normalize_level(decision.get("final_level", "A1"))
            return BenchmarkResult(system, "diagnosis", scenario["name"], True, "", scenario["gold_level"], detected, "", trace, [], [], [], [], _empty_retrieval_stats(), {})
    except Exception as exc:
        return BenchmarkResult(system, "diagnosis", scenario["name"], False, str(exc), scenario["gold_level"], "", "", [], [], [], [], [], _empty_retrieval_stats(), {})


def run_diagnosis_lexi(scenario: Dict[str, Any]) -> BenchmarkResult:
    system = "Lexi_Path_German"
    try:
        with project_context(ROOT / "LexiPath_German"):
            purge_modules()
            app_mod = importlib.import_module("app")
            learner_store = importlib.import_module("learner_store")

            learner_id = f"diag_{scenario['name']}"
            learner_store.delete_learner(learner_id)
            state = app_mod.build_initial_state(learner_id, learner_id)
            state = _lexi_invoke_with_timeout(app_mod.app, state)
            trace: List[Dict[str, Any]] = []

            for _ in range(20):
                if state.get("phase") == "tutoring":
                    break
                current_id = state.get("diagnostic_id", 0)
                task = app_mod.DiagnosticManager.get_task(current_id)
                if not task:
                    break
                prompt = state["messages"][-1]["content"] if state.get("messages") else ""
                answer = simulate_learner_answer(prompt, scenario["learner_profile"], task["level"], task["grammar_point"])
                state["messages"].append({"role": "user", "content": answer})
                state = _lexi_invoke_with_timeout(app_mod.app, state)
                trace.append(
                    {
                        "task_id": current_id,
                        "target_level": task["level"],
                        "topic": task["topic"],
                        "grammar_point": task["grammar_point"],
                        "question": _extract_visible_question(prompt),
                        "learner_answer": answer,
                        "score_value": state.get("diagnostic_results", {}).get(current_id, 0),
                    }
                )

            detected = _normalize_level(state.get("user_level", "A1"))
            return BenchmarkResult(system, "diagnosis", scenario["name"], True, "", scenario["gold_level"], detected, "", trace, [], [], [], [], _empty_retrieval_stats(), {})
    except Exception as exc:
        return BenchmarkResult(system, "diagnosis", scenario["name"], False, str(exc), scenario["gold_level"], "", "", [], [], [], [], [], _empty_retrieval_stats(), {})


def run_tutoring_zeroshot(suite: str, scenario: Dict[str, Any]) -> BenchmarkResult:
    system = "Zeroshot"
    try:
        with project_context(ROOT / "Zeroshot"):
            purge_modules()
            engine = importlib.import_module("engine")
            prompts = importlib.import_module("prompts")
            syllabus = importlib.import_module("syllabus")
            tutor = importlib.import_module("tutor")

            level = scenario["level"]
            compact_plan = syllabus.format_compact_learning_path(level)
            choice = "focus on speaking"
            next_level = tutor._get_next_level(level)
            starter_message = f"""
The learner has completed a diagnosis.

Detected level: {level}
Strengths: {_minimal_summary(level)['strengths']}
Weaknesses: {_minimal_summary(level)['weaknesses']}

Compact learning path:
{compact_plan}

Learner preference:
{tutor.TUTOR_START_OPTIONS[choice]}
""".strip()

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": starter_message},
            ]
            session1_assistant: List[str] = []
            first = engine.call_kimi(messages)
            messages.append({"role": "assistant", "content": first})
            session1_assistant.append(first)

            for user_msg in scenario["messages"]:
                messages.append({"role": "user", "content": user_msg})
                reply = engine.call_kimi(messages)
                messages.append({"role": "assistant", "content": reply})
                session1_assistant.append(reply)

            metrics = _metrics_from_sessions(session1_assistant, [], [])
            return BenchmarkResult(system, suite, scenario["name"], True, "", "", "", level, [], list(scenario["messages"]), session1_assistant, [], [], _empty_retrieval_stats(), metrics)
    except Exception as exc:
        return BenchmarkResult(system, suite, scenario["name"], False, str(exc), "", "", scenario["level"], [], [], [], [], [], _empty_retrieval_stats(), {})


def _llmrag_bundle(level: str, user_message: str, rag: Any, tutor: Any) -> Dict[str, Any]:
    query = tutor.build_retrieval_query(level, user_message)
    preferred_level = None if level == "Pre-A1" else level
    chunks = rag.retrieve(query, level_filter=preferred_level)
    used_fallback = False
    if not chunks and preferred_level is not None:
        chunks = rag.retrieve(query, level_filter=None)
        used_fallback = bool(chunks)
    return {"chunks": chunks, "context": rag.format_context(chunks), "used_fallback": used_fallback}


def run_tutoring_llmrag(suite: str, scenario: Dict[str, Any]) -> BenchmarkResult:
    system = "llmrag"
    try:
        with project_context(ROOT / "llmrag"):
            purge_modules()
            tutor = importlib.import_module("tutor")
            rag = importlib.import_module("rag")
            prompts = importlib.import_module("prompts")
            engine = importlib.import_module("engine")

            level = scenario["level"]
            stats = _empty_retrieval_stats()
            starter_bundle = _llmrag_bundle(level, "Start teaching me now with a suitable first lesson.", rag, tutor)
            stats["turns"] += 1
            stats["total_chunks"] += len(starter_bundle["chunks"])
            if starter_bundle["chunks"]:
                stats["turns_with_chunks"] += 1
            if starter_bundle["used_fallback"]:
                stats["fallback_turns"] += 1

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": prompts.build_tutor_starter_prompt(level, starter_bundle["context"])},
            ]
            session1_assistant: List[str] = []
            first = engine.call_chat(messages)
            messages.append({"role": "assistant", "content": first})
            session1_assistant.append(first)

            for user_msg in scenario["messages"]:
                bundle = _llmrag_bundle(level, user_msg, rag, tutor)
                stats["turns"] += 1
                stats["total_chunks"] += len(bundle["chunks"])
                if bundle["chunks"]:
                    stats["turns_with_chunks"] += 1
                if bundle["used_fallback"]:
                    stats["fallback_turns"] += 1
                messages.append({"role": "user", "content": f"Retrieved context:\n{bundle['context']}\n\nLearner says: {user_msg}"})
                reply = engine.call_chat(messages)
                messages.append({"role": "assistant", "content": reply})
                session1_assistant.append(reply)

            metrics = _metrics_from_sessions(session1_assistant, [], [])
            return BenchmarkResult(system, suite, scenario["name"], True, "", "", "", level, [], list(scenario["messages"]), session1_assistant, [], [], stats, metrics)
    except Exception as exc:
        return BenchmarkResult(system, suite, scenario["name"], False, str(exc), "", "", scenario["level"], [], [], [], [], [], _empty_retrieval_stats(), {})


def run_tutoring_lexi(suite: str, scenario: Dict[str, Any]) -> BenchmarkResult:
    system = "Lexi_Path_German"
    try:
        with project_context(ROOT / "LexiPath_German"):
            purge_modules()
            app_mod = importlib.import_module("app")

            learner_id = f"{suite}_{scenario['name']}"
            saved = {
                "display_name": learner_id,
                "user_level": scenario["level"],
                "diagnostic_results": {},
                "diagnostic_feedback": [],
                "learner_profile": app_mod.default_learner_profile(),
                "grammar_point_mastery": app_mod.default_grammar_point_mastery(),
                "level_source": "benchmark_seed",
                "level_confidence": "high",
            }
            state = app_mod.build_state_from_saved_learner(learner_id, saved)
            session1_assistant: List[str] = []
            stats = _empty_retrieval_stats()

            for user_msg in scenario["messages"]:
                state["messages"].append({"role": "user", "content": user_msg})
                state = _lexi_invoke_with_timeout(app_mod.app, state)
                session1_assistant.append(state["messages"][-1]["content"])
                docs = state.get("retrieved_documents", []) or []
                stats["turns"] += 1
                stats["total_chunks"] += len(docs)
                if docs:
                    stats["turns_with_chunks"] += 1
                if state.get("retrieval_used_fallback", False):
                    stats["fallback_turns"] += 1

            metrics = _metrics_from_sessions(session1_assistant, [], [])
            return BenchmarkResult(system, suite, scenario["name"], True, "", "", "", scenario["level"], [], list(scenario["messages"]), session1_assistant, [], [], stats, metrics)
    except Exception as exc:
        return BenchmarkResult(system, suite, scenario["name"], False, str(exc), "", "", scenario["level"], [], [], [], [], [], _empty_retrieval_stats(), {})


def run_continuity_zeroshot(scenario: Dict[str, Any]) -> BenchmarkResult:
    system = "Zeroshot"
    try:
        with project_context(ROOT / "Zeroshot"):
            purge_modules()
            engine = importlib.import_module("engine")
            prompts = importlib.import_module("prompts")
            syllabus = importlib.import_module("syllabus")

            level = scenario["level"]
            compact_plan = syllabus.format_compact_learning_path(level)
            starter = f"Detected level: {level}\n\nCompact learning path:\n{compact_plan}"
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": starter},
            ]
            session1_assistant: List[str] = []
            first = engine.call_kimi(messages)
            messages.append({"role": "assistant", "content": first})
            session1_assistant.append(first)
            for user_msg in scenario["session1_messages"]:
                messages.append({"role": "user", "content": user_msg})
                reply = engine.call_kimi(messages)
                messages.append({"role": "assistant", "content": reply})
                session1_assistant.append(reply)

            messages2: List[Dict[str, str]] = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": f"The learner is returning at level {level}. Continue naturally."},
            ]
            session2_assistant: List[str] = []
            first2 = engine.call_kimi(messages2)
            messages2.append({"role": "assistant", "content": first2})
            session2_assistant.append(first2)
            for user_msg in scenario["session2_messages"]:
                messages2.append({"role": "user", "content": user_msg})
                reply = engine.call_kimi(messages2)
                messages2.append({"role": "assistant", "content": reply})
                session2_assistant.append(reply)

            metrics = _metrics_from_sessions(session1_assistant, session2_assistant, scenario["memory_keywords"])
            return BenchmarkResult(system, "continuity", scenario["name"], True, "", "", "", level, [], list(scenario["session1_messages"]), session1_assistant, list(scenario["session2_messages"]), session2_assistant, _empty_retrieval_stats(), metrics)
    except Exception as exc:
        return BenchmarkResult(system, "continuity", scenario["name"], False, str(exc), "", "", scenario["level"], [], [], [], [], [], _empty_retrieval_stats(), {})


def run_continuity_llmrag(scenario: Dict[str, Any]) -> BenchmarkResult:
    system = "llmrag"
    try:
        with project_context(ROOT / "llmrag"):
            purge_modules()
            tutor = importlib.import_module("tutor")
            rag = importlib.import_module("rag")
            prompts = importlib.import_module("prompts")
            engine = importlib.import_module("engine")

            level = scenario["level"]
            stats = _empty_retrieval_stats()
            bundle = _llmrag_bundle(level, "Start teaching me now with a suitable first lesson.", rag, tutor)
            stats["turns"] += 1
            stats["total_chunks"] += len(bundle["chunks"])
            if bundle["chunks"]:
                stats["turns_with_chunks"] += 1
            if bundle["used_fallback"]:
                stats["fallback_turns"] += 1

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": prompts.build_tutor_starter_prompt(level, bundle["context"])},
            ]
            session1_assistant: List[str] = []
            first = engine.call_chat(messages)
            messages.append({"role": "assistant", "content": first})
            session1_assistant.append(first)
            for user_msg in scenario["session1_messages"]:
                current = _llmrag_bundle(level, user_msg, rag, tutor)
                stats["turns"] += 1
                stats["total_chunks"] += len(current["chunks"])
                if current["chunks"]:
                    stats["turns_with_chunks"] += 1
                if current["used_fallback"]:
                    stats["fallback_turns"] += 1
                messages.append({"role": "user", "content": f"Retrieved context:\n{current['context']}\n\nLearner says: {user_msg}"})
                reply = engine.call_chat(messages)
                messages.append({"role": "assistant", "content": reply})
                session1_assistant.append(reply)

            bundle2 = _llmrag_bundle(level, "This is a new session. Continue from my level.", rag, tutor)
            stats["turns"] += 1
            stats["total_chunks"] += len(bundle2["chunks"])
            if bundle2["chunks"]:
                stats["turns_with_chunks"] += 1
            if bundle2["used_fallback"]:
                stats["fallback_turns"] += 1
            messages2: List[Dict[str, str]] = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": f"The learner is returning in a new session.\n\n{prompts.build_tutor_starter_prompt(level, bundle2['context'])}"},
            ]
            session2_assistant: List[str] = []
            first2 = engine.call_chat(messages2)
            messages2.append({"role": "assistant", "content": first2})
            session2_assistant.append(first2)
            for user_msg in scenario["session2_messages"]:
                current = _llmrag_bundle(level, user_msg, rag, tutor)
                stats["turns"] += 1
                stats["total_chunks"] += len(current["chunks"])
                if current["chunks"]:
                    stats["turns_with_chunks"] += 1
                if current["used_fallback"]:
                    stats["fallback_turns"] += 1
                messages2.append({"role": "user", "content": f"Retrieved context:\n{current['context']}\n\nLearner says: {user_msg}"})
                reply = engine.call_chat(messages2)
                messages2.append({"role": "assistant", "content": reply})
                session2_assistant.append(reply)

            metrics = _metrics_from_sessions(session1_assistant, session2_assistant, scenario["memory_keywords"])
            return BenchmarkResult(system, "continuity", scenario["name"], True, "", "", "", level, [], list(scenario["session1_messages"]), session1_assistant, list(scenario["session2_messages"]), session2_assistant, stats, metrics)
    except Exception as exc:
        return BenchmarkResult(system, "continuity", scenario["name"], False, str(exc), "", "", scenario["level"], [], [], [], [], [], _empty_retrieval_stats(), {})


def run_continuity_lexi(scenario: Dict[str, Any]) -> BenchmarkResult:
    system = "Lexi_Path_German"
    try:
        with project_context(ROOT / "LexiPath_German"):
            purge_modules()
            app_mod = importlib.import_module("app")
            learner_store = importlib.import_module("learner_store")

            learner_id = scenario["learner_id"]
            learner_store.delete_learner(learner_id)
            saved = {
                "display_name": scenario["display_name"],
                "user_level": scenario["level"],
                "diagnostic_results": {},
                "diagnostic_feedback": [],
                "learner_profile": app_mod.default_learner_profile(),
                "grammar_point_mastery": app_mod.default_grammar_point_mastery(),
                "level_source": "benchmark_seed",
                "level_confidence": "high",
            }
            state = app_mod.build_state_from_saved_learner(learner_id, saved)
            session1_assistant: List[str] = []
            stats = _empty_retrieval_stats()

            for user_msg in scenario["session1_messages"]:
                state["messages"].append({"role": "user", "content": user_msg})
                state = _lexi_invoke_with_timeout(app_mod.app, state)
                session1_assistant.append(state["messages"][-1]["content"])
                docs = state.get("retrieved_documents", []) or []
                stats["turns"] += 1
                stats["total_chunks"] += len(docs)
                if docs:
                    stats["turns_with_chunks"] += 1
                if state.get("retrieval_used_fallback", False):
                    stats["fallback_turns"] += 1

            saved_after = learner_store.load_learner(learner_id) or saved
            state2 = app_mod.build_state_from_saved_learner(learner_id, saved_after)
            session2_assistant: List[str] = []
            welcome = (
                f"Welcome back, {saved_after.get('display_name', scenario['display_name'])}. "
                f"I remember you around {saved_after.get('user_level', scenario['level'])}. "
                "What would you like to work on today?"
            )
            state2["messages"].append({"role": "assistant", "content": welcome})
            session2_assistant.append(welcome)

            for user_msg in scenario["session2_messages"]:
                state2["messages"].append({"role": "user", "content": user_msg})
                state2 = _lexi_invoke_with_timeout(app_mod.app, state2)
                session2_assistant.append(state2["messages"][-1]["content"])
                docs = state2.get("retrieved_documents", []) or []
                stats["turns"] += 1
                stats["total_chunks"] += len(docs)
                if docs:
                    stats["turns_with_chunks"] += 1
                if state2.get("retrieval_used_fallback", False):
                    stats["fallback_turns"] += 1

            metrics = _metrics_from_sessions(session1_assistant, session2_assistant, scenario["memory_keywords"])
            return BenchmarkResult(system, "continuity", scenario["name"], True, "", "", "", scenario["level"], [], list(scenario["session1_messages"]), session1_assistant, list(scenario["session2_messages"]), session2_assistant, stats, metrics)
    except Exception as exc:
        return BenchmarkResult(system, "continuity", scenario["name"], False, str(exc), "", "", scenario["level"], [], [], [], [], [], _empty_retrieval_stats(), {})


def _result_to_json(result: BenchmarkResult) -> Dict[str, Any]:
    return {
        "system": result.system_name,
        "suite": result.suite,
        "scenario_name": result.scenario_name,
        "success": result.success,
        "error": result.error,
        "gold_level": result.gold_level,
        "detected_level": result.detected_level,
        "tutor_level": result.tutor_level,
        "diagnostic_trace": result.diagnostic_trace,
        "session1_user": result.session1_user,
        "session1_assistant": result.session1_assistant,
        "session2_user": result.session2_user,
        "session2_assistant": result.session2_assistant,
        "retrieval_stats": result.retrieval_stats,
        "metrics": result.metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation suites for Zeroshot, llmrag, and Lexi_Path_German")
    parser.add_argument("--output", default="benchmark_report.json", help="Output benchmark report path")
    parser.add_argument("--quick", action="store_true", help="Run one scenario per suite for a faster smoke test")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=["zeroshot", "llmrag", "lexi"],
        choices=["zeroshot", "llmrag", "lexi"],
        help="Subset of systems to evaluate",
    )
    args = parser.parse_args()

    diagnosis_runners = {
        "zeroshot": run_diagnosis_zeroshot,
        "llmrag": run_diagnosis_llmrag,
        "lexi": run_diagnosis_lexi,
    }
    tutoring_runners = {
        "zeroshot": run_tutoring_zeroshot,
        "llmrag": run_tutoring_llmrag,
        "lexi": run_tutoring_lexi,
    }
    continuity_runners = {
        "zeroshot": run_continuity_zeroshot,
        "llmrag": run_continuity_llmrag,
        "lexi": run_continuity_lexi,
    }

    diagnosis_scenarios = DIAGNOSIS_SCENARIOS[:1] if args.quick else DIAGNOSIS_SCENARIOS
    tutoring_scenarios = TUTORING_SCENARIOS[:1] if args.quick else TUTORING_SCENARIOS
    continuity_scenarios = CONTINUITY_SCENARIOS[:1] if args.quick else CONTINUITY_SCENARIOS

    started = time.time()
    results: List[BenchmarkResult] = []

    for system in args.systems:
        print(f"Running {system} diagnosis benchmark...")
        for scenario in diagnosis_scenarios:
            results.append(_with_retries(diagnosis_runners[system], scenario))

        print(f"Running {system} tutoring benchmark...")
        for scenario in tutoring_scenarios:
            results.append(_with_retries(tutoring_runners[system], "tutoring", scenario))

        print(f"Running {system} continuity benchmark...")
        for scenario in continuity_scenarios:
            results.append(_with_retries(continuity_runners[system], scenario))

    payload = {
        "generated_at_epoch": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
        "quick_mode": args.quick,
        "scenario_catalog": {
            "diagnosis": diagnosis_scenarios,
            "tutoring": tutoring_scenarios,
            "continuity": continuity_scenarios,
        },
        "results": [_result_to_json(item) for item in results],
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved benchmark report: {out_path}")


if __name__ == "__main__":
    main()
