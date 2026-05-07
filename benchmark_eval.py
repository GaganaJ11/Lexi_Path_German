import argparse
import importlib
import json
import os
import re
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

BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        "id": "diag_a1_foundation",
        "track": "diagnostic",
        "gold_level": "A1",
        "learner_profile": {
            "true_level": "A1",
            "strengths": ["negation_kein"],
            "weaknesses": ["perfect_tense_basics", "subordinate_clause_weil"],
            "style": "short",
            "noise": "light",
        },
    },
    {
        "id": "diag_a2_everyday",
        "track": "diagnostic",
        "gold_level": "A2",
        "learner_profile": {
            "true_level": "A2",
            "strengths": ["perfect_tense_basics", "comparatives_basics"],
            "weaknesses": ["subordinate_clause_weil"],
            "style": "short",
            "noise": "light",
        },
    },
    {
        "id": "diag_b1_work",
        "track": "diagnostic",
        "gold_level": "B1",
        "learner_profile": {
            "true_level": "B1",
            "strengths": ["subordinate_clause_weil", "relative_clauses_basics"],
            "weaknesses": ["konjunktiv_ii_basics"],
            "style": "short",
            "noise": "light",
        },
    },
    {
        "id": "teach_explain_weil_a2",
        "track": "tutoring",
        "start_level": "A2",
        "prompts": [
            "Please explain weil clauses simply and give one short example.",
            "Now give me one short practice question.",
        ],
        "expected_keywords": ["weil", "verb", "example"],
        "expect_exercise": True,
    },
    {
        "id": "teach_correction_perfekt_a2",
        "track": "tutoring",
        "start_level": "A2",
        "prompts": [
            "Please correct this sentence and explain it simply: Ich habe gestern Deutsch lernen.",
        ],
        "expected_keywords": ["habe", "gelernt"],
        "corrected_form": "Ich habe gestern Deutsch gelernt.",
    },
    {
        "id": "teach_study_plan_b1",
        "track": "tutoring",
        "start_level": "B1",
        "prompts": [
            "Make me a short study plan to improve German for work communication.",
        ],
        "expected_keywords": ["work", "plan", "practice"],
        "expect_study_plan": True,
    },
    {
        "id": "continuity_returning_perfekt",
        "track": "continuity",
        "gold_level": "A2",
        "learner_profile": {
            "true_level": "A2",
            "strengths": ["perfect_tense_basics"],
            "weaknesses": ["subordinate_clause_weil"],
            "style": "short",
            "noise": "light",
        },
        "session1_messages": [
            "I want help with Perfekt for daily conversation.",
            "Please explain Perfekt and give me one small exercise.",
        ],
        "session2_messages": [
            "Can we continue from where we stopped yesterday?",
            "Give me another short exercise now.",
        ],
        "memory_keywords": ["perfekt", "daily", "exercise", "a2"],
    },
    {
        "id": "continuity_returning_work_b1",
        "track": "continuity",
        "gold_level": "B1",
        "learner_profile": {
            "true_level": "B1",
            "strengths": ["subordinate_clause_weil", "relative_clauses_basics"],
            "weaknesses": ["konjunktiv_ii_basics"],
            "style": "short",
            "noise": "light",
        },
        "session1_messages": [
            "I want to improve German for work communication.",
            "Please help me with weil clauses for speaking at work.",
        ],
        "session2_messages": [
            "Can we continue from yesterday?",
            "Give me one short speaking exercise.",
        ],
        "memory_keywords": ["weil", "work", "speaking", "b1"],
    },
]


@dataclass
class BenchmarkResult:
    system: str
    case_id: str
    track: str
    success: bool
    error: str
    detected_level: str
    assistant_outputs: List[str]
    user_inputs: List[str]
    diagnostic_trace: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]
    signals: Dict[str, Any]


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
    return {"turns": 0, "turns_with_chunks": 0, "total_chunks": 0, "fallback_turns": 0}


def _normalize_level(level: str) -> str:
    if level in {"A1", "A2", "B1"}:
        return level
    if level == "Pre-A1":
        return "A1"
    return "A1"


def _clamp_level(level: str) -> str:
    return level if level in LEVEL_RANK else "A1"


def _rank(level: str) -> int:
    return LEVEL_RANK.get(_clamp_level(level), 1)


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


def _choose_quality(profile: Dict[str, Any], target_level: str, grammar_point: str) -> str:
    true_level = _clamp_level(profile.get("true_level", "A1"))
    strengths = set(profile.get("strengths", []))
    weaknesses = set(profile.get("weaknesses", []))
    score = _rank(true_level) - _rank(target_level)
    if grammar_point in strengths:
        score += 1
    if grammar_point in weaknesses:
        score -= 1
    if score >= 0:
        return "full"
    if score == -1:
        return "partial"
    return "fail"


def _answer_bank(grammar_point: str, quality: str, question: str) -> str:
    bank = {
        "basic_greeting_phrase": {"full": "Hallo, ich heiße Anna.", "partial": "Hallo ich Anna.", "fail": "Hallo."},
        "yes_no_basic_response": {"full": "Ja, ein bisschen.", "partial": "Ja bisschen.", "fail": "Ja."},
        "single_word_everyday_vocab": {"full": "Wasser.", "partial": "Wasser trinken.", "fail": "Essen."},
        "indefinite_articles_ein_eine_einen": {"full": "Der Mann isst einen Apfel.", "partial": "Der Mann isst ein Apfel.", "fail": "Der Mann isst Apfel."},
        "negation_kein": {"full": "Ich habe kein Auto.", "partial": "Ich habe nicht Auto.", "fail": "Ich habe Auto."},
        "present_tense_basic_verbs": {"full": "Ich wohne in Berlin.", "partial": "Ich wohnen in Berlin.", "fail": "Berlin."},
        "perfect_tense_basics": {"full": "Ich habe gestern Deutsch gelernt.", "partial": "Ich lerne gestern Deutsch.", "fail": "Ich lerne Deutsch."},
        "accusative_with_movement": {"full": "Ich lege das Buch auf den Tisch.", "partial": "Ich lege das Buch auf dem Tisch.", "fail": "Das Buch ist auf dem Tisch."},
        "comparatives_basics": {"full": "Ein Auto ist schneller als ein Fahrrad.", "partial": "Ein Auto ist mehr schnell als ein Fahrrad.", "fail": "Auto und Fahrrad sind gut."},
        "subordinate_clause_weil": {"full": "Ich lerne Deutsch, weil ich in Deutschland arbeiten will.", "partial": "Ich lerne Deutsch, weil ich will in Deutschland arbeiten.", "fail": "Ich lerne Deutsch. Ich will in Deutschland arbeiten."},
        "konjunktiv_ii_basics": {"full": "Ich würde mehr reisen, wenn ich Zeit hätte.", "partial": "Ich werde mehr reisen, wenn ich Zeit hätte.", "fail": "Ich reise mehr."},
        "relative_clauses_basics": {"full": "Das ist die Frau, die mir hilft.", "partial": "Das ist die Frau die hilft mir.", "fail": "Das ist eine Frau. Sie hilft mir."},
    }
    if grammar_point in bank:
        return bank[grammar_point][quality]
    lowered = question.lower()
    if "weil" in lowered:
        return bank["subordinate_clause_weil"][quality]
    if "perfekt" in lowered or "yesterday" in lowered or "past" in lowered:
        return bank["perfect_tense_basics"][quality]
    if "als" in lowered or "compare" in lowered:
        return bank["comparatives_basics"][quality]
    if "relative" in lowered:
        return bank["relative_clauses_basics"][quality]
    fallback = {"full": "Ich lerne Deutsch jeden Tag.", "partial": "Ich lernen Deutsch jeden Tag.", "fail": "Deutsch jeden Tag."}
    return fallback[quality]


def simulate_learner_answer(question_text: str, profile: Dict[str, Any], target_level: str, grammar_point: str) -> str:
    visible = _extract_visible_question(question_text)
    quality = _choose_quality(profile, target_level, grammar_point)
    return _answer_bank(grammar_point, quality, visible)


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _question_like(text: str) -> bool:
    lowered = text.lower()
    return "?" in text or any(word in lowered for word in ["exercise", "try", "write", "practice", "question"])


def _study_plan_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    structured = sum(
        1
        for line in lines
        if line.startswith("-") or line.startswith("1.") or line.startswith("2.") or line.startswith("3.")
    )
    return structured >= 2


def score_tutoring_signals(case: Dict[str, Any], outputs: List[str]) -> Dict[str, Any]:
    joined = "\n".join(outputs)
    signals = {
        "assistant_turns": len(outputs),
        "avg_response_chars": round(sum(len(text) for text in outputs) / len(outputs), 1) if outputs else 0.0,
        "keyword_hit": _contains_any(joined, case.get("expected_keywords", [])),
        "keyword_hit_count": sum(1 for keyword in case.get("expected_keywords", []) if keyword.lower() in joined.lower()),
        "exercise_offer": any(_question_like(text) for text in outputs),
        "study_plan_structure": any(_study_plan_like(text) for text in outputs),
        "correction_hit": False,
    }
    corrected_form = case.get("corrected_form")
    if corrected_form:
        signals["correction_hit"] = corrected_form.lower() in joined.lower()
    return signals


def score_continuity_signals(case: Dict[str, Any], session1: List[str], session2: List[str]) -> Dict[str, Any]:
    all_replies = session1 + session2
    session2_blob = "\n".join(session2).lower()
    keywords = case.get("memory_keywords", [])
    hits = sum(1 for keyword in keywords if keyword.lower() in session2_blob)
    return {
        "assistant_turns": len(all_replies),
        "avg_response_chars": round(sum(len(text) for text in all_replies) / len(all_replies), 1) if all_replies else 0.0,
        "followup_question_rate": round(sum(1 for text in all_replies if "?" in text) / len(all_replies), 3) if all_replies else 0.0,
        "session2_memory_recall": hits > 0,
        "session2_memory_keyword_hits": hits,
    }


def _is_transient_error(message: str) -> bool:
    lowered = (message or "").lower()
    return any(marker in lowered for marker in ["502", "503", "504", "bad gateway", "timeout", "temporarily unavailable"])


def run_with_retries(runner: Callable[[Dict[str, Any]], BenchmarkResult], case: Dict[str, Any], attempts: int = 3) -> BenchmarkResult:
    last: Optional[BenchmarkResult] = None
    for idx in range(attempts):
        result = runner(case)
        if result.success:
            return result
        last = result
        if idx >= attempts - 1 or not _is_transient_error(result.error):
            return result
        time.sleep(2.0 * (idx + 1))
    return last if last is not None else runner(case)


def run_zeroshot_diagnostic(case: Dict[str, Any]) -> BenchmarkResult:
    try:
        with project_context(ROOT / "Zeroshot"):
            purge_modules()
            diagnostic_logic = importlib.import_module("diagnostic_logic")

            trace: List[Dict[str, Any]] = []
            results_by_task: Dict[int, int] = {}
            current_id = diagnostic_logic.DiagnosticManager.get_start_task_id()
            current_level = diagnostic_logic.DiagnosticManager.get_level_for_task(current_id) if current_id is not None else "A1"

            while current_id is not None:
                task = diagnostic_logic.DiagnosticManager.get_task(current_id)
                if not task:
                    break
                question = diagnostic_logic.generate_diagnostic_question(task, user_level=current_level)
                answer = simulate_learner_answer(question, case["learner_profile"], task["level"], task["grammar_point"])
                grade = diagnostic_logic.grade_diagnostic_answer(task, answer)
                trace.append(
                    {
                        "task_id": task["id"],
                        "target_level": task["level"],
                        "topic": task["topic"],
                        "grammar_point": task["grammar_point"],
                        "question": question,
                        "learner_answer": answer,
                        "score_value": grade["score_value"],
                    }
                )
                results_by_task[current_id] = grade["score_value"]
                next_id = diagnostic_logic.DiagnosticManager.get_next_task_id(current_id, results_by_task)
                if next_id is not None:
                    current_level = diagnostic_logic.DiagnosticManager.get_level_for_task(next_id)
                current_id = next_id

            detected_level = _normalize_level(diagnostic_logic.DiagnosticManager.determine_final_level(results_by_task))
            over = _rank(detected_level) > _rank(case["gold_level"])
            under = _rank(detected_level) < _rank(case["gold_level"])
            return BenchmarkResult(
                system="Zeroshot",
                case_id=case["id"],
                track="diagnostic",
                success=True,
                error="",
                detected_level=detected_level,
                assistant_outputs=[],
                user_inputs=[],
                diagnostic_trace=trace,
                retrieval_stats=_empty_retrieval_stats(),
                signals={"turns": len(trace), "over_placement": over, "under_placement": under},
            )
    except Exception as exc:
        return BenchmarkResult("Zeroshot", case["id"], "diagnostic", False, str(exc), "", [], [], [], _empty_retrieval_stats(), {})


def run_llmrag_diagnostic(case: Dict[str, Any]) -> BenchmarkResult:
    try:
        with project_context(ROOT / "llmrag"):
            purge_modules()
            diagnostic_logic = importlib.import_module("diagnostic_logic")

            attempts: List[Dict[str, Any]] = []
            trace: List[Dict[str, Any]] = []
            decision = {"final_level": "A1", "action": "continue"}

            while len(attempts) < getattr(diagnostic_logic, "MAX_ATTEMPTS", 6):
                plan = diagnostic_logic._plan_next_question(attempts)
                question = diagnostic_logic._generate_question(plan, attempts)
                answer = simulate_learner_answer(question, case["learner_profile"], plan["target_level"], plan["grammar_point"])
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

            detected_level = _normalize_level(decision.get("final_level", "A1"))
            return BenchmarkResult(
                system="llmrag",
                case_id=case["id"],
                track="diagnostic",
                success=True,
                error="",
                detected_level=detected_level,
                assistant_outputs=[],
                user_inputs=[],
                diagnostic_trace=trace,
                retrieval_stats=_empty_retrieval_stats(),
                signals={
                    "turns": len(trace),
                    "over_placement": _rank(detected_level) > _rank(case["gold_level"]),
                    "under_placement": _rank(detected_level) < _rank(case["gold_level"]),
                },
            )
    except Exception as exc:
        return BenchmarkResult("llmrag", case["id"], "diagnostic", False, str(exc), "", [], [], [], _empty_retrieval_stats(), {})


def run_lexi_diagnostic(case: Dict[str, Any]) -> BenchmarkResult:
    learner_id = f"benchmark_{case['id']}"
    try:
        with project_context(ROOT / "LexiPath_German"):
            purge_modules()
            app_mod = importlib.import_module("app")
            learner_store = importlib.import_module("learner_store")

            learner_store.delete_learner(learner_id)
            state = app_mod.build_initial_state(learner_id, "Benchmark User")
            state = app_mod.app.invoke(state)
            trace: List[Dict[str, Any]] = []

            for _ in range(20):
                if state.get("phase") == "tutoring":
                    break
                current_id = state.get("diagnostic_id", 0)
                task = app_mod.DiagnosticManager.get_task(current_id)
                if not task:
                    break
                assistant_text = state["messages"][-1]["content"] if state.get("messages") else ""
                answer = simulate_learner_answer(assistant_text, case["learner_profile"], task["level"], task["grammar_point"])
                state["messages"].append({"role": "user", "content": answer})
                state = app_mod.app.invoke(state)
                trace.append(
                    {
                        "task_id": current_id,
                        "target_level": task["level"],
                        "topic": task["topic"],
                        "grammar_point": task["grammar_point"],
                        "question": _extract_visible_question(assistant_text),
                        "learner_answer": answer,
                        "score_value": state.get("diagnostic_results", {}).get(current_id, 0),
                    }
                )

            detected_level = _normalize_level(state.get("user_level", "A1"))
            learner_store.delete_learner(learner_id)
            return BenchmarkResult(
                system="Lexi_Path_German",
                case_id=case["id"],
                track="diagnostic",
                success=True,
                error="",
                detected_level=detected_level,
                assistant_outputs=[],
                user_inputs=[],
                diagnostic_trace=trace,
                retrieval_stats=_empty_retrieval_stats(),
                signals={
                    "turns": len(trace),
                    "over_placement": _rank(detected_level) > _rank(case["gold_level"]),
                    "under_placement": _rank(detected_level) < _rank(case["gold_level"]),
                },
            )
    except Exception as exc:
        return BenchmarkResult("Lexi_Path_German", case["id"], "diagnostic", False, str(exc), "", [], [], [], _empty_retrieval_stats(), {})


def run_zeroshot_tutoring(case: Dict[str, Any]) -> BenchmarkResult:
    try:
        with project_context(ROOT / "Zeroshot"):
            purge_modules()
            engine = importlib.import_module("engine")
            prompts = importlib.import_module("prompts")
            syllabus = importlib.import_module("syllabus")
            tutor = importlib.import_module("tutor")

            level = case["start_level"]
            compact_plan = syllabus.format_compact_learning_path(level)
            next_level = tutor._get_next_level(level)
            starter_message = f"""
The learner has completed a diagnosis.

Detected level: {level}
Compact learning path:
{compact_plan}

Learner preference:
{tutor.TUTOR_START_OPTIONS['focus on speaking']}

Please act as a warm, human-like private German tutor.
Start directly with a suitable first topic and keep the lesson practical.
Mention that once {level} is complete, continue with {next_level}.
""".strip()

            messages = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": starter_message},
            ]
            outputs: List[str] = []
            first = engine.call_kimi(messages)
            messages.append({"role": "assistant", "content": first})
            outputs.append(first)
            for prompt in case["prompts"]:
                messages.append({"role": "user", "content": prompt})
                reply = engine.call_kimi(messages)
                messages.append({"role": "assistant", "content": reply})
                outputs.append(reply)
            return BenchmarkResult("Zeroshot", case["id"], "tutoring", True, "", level, outputs, list(case["prompts"]), [], _empty_retrieval_stats(), score_tutoring_signals(case, outputs))
    except Exception as exc:
        return BenchmarkResult("Zeroshot", case["id"], "tutoring", False, str(exc), "", [], [], [], _empty_retrieval_stats(), {})


def _llmrag_retrieve_bundle(level: str, user_message: str, rag: Any, tutor: Any) -> Dict[str, Any]:
    query = tutor.build_retrieval_query(level, user_message)
    preferred_level = None if level == "Pre-A1" else level
    chunks = rag.retrieve(query, level_filter=preferred_level)
    used_fallback = False
    if not chunks and preferred_level is not None:
        chunks = rag.retrieve(query, level_filter=None)
        used_fallback = bool(chunks)
    return {"chunks": chunks, "context": rag.format_context(chunks), "used_fallback": used_fallback}


def run_llmrag_tutoring(case: Dict[str, Any]) -> BenchmarkResult:
    try:
        with project_context(ROOT / "llmrag"):
            purge_modules()
            engine = importlib.import_module("engine")
            prompts = importlib.import_module("prompts")
            rag = importlib.import_module("rag")
            tutor = importlib.import_module("tutor")

            level = case["start_level"]
            retrieval_stats = _empty_retrieval_stats()
            start_request = "Start teaching me now with a suitable first lesson."
            bundle = _llmrag_retrieve_bundle(level, start_request, rag, tutor)
            retrieval_stats["turns"] += 1
            retrieval_stats["total_chunks"] += len(bundle["chunks"])
            if bundle["chunks"]:
                retrieval_stats["turns_with_chunks"] += 1
            if bundle["used_fallback"]:
                retrieval_stats["fallback_turns"] += 1

            starter = prompts.build_tutor_starter_prompt(level, bundle["context"])
            messages = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": starter},
            ]
            outputs: List[str] = []
            first = engine.call_chat(messages)
            messages.append({"role": "assistant", "content": first})
            outputs.append(first)

            for prompt in case["prompts"]:
                bundle = _llmrag_retrieve_bundle(level, prompt, rag, tutor)
                retrieval_stats["turns"] += 1
                retrieval_stats["total_chunks"] += len(bundle["chunks"])
                if bundle["chunks"]:
                    retrieval_stats["turns_with_chunks"] += 1
                if bundle["used_fallback"]:
                    retrieval_stats["fallback_turns"] += 1
                messages.append({"role": "user", "content": f"Retrieved context:\n{bundle['context']}\n\nLearner says: {prompt}"})
                reply = engine.call_chat(messages)
                messages.append({"role": "assistant", "content": reply})
                outputs.append(reply)

            return BenchmarkResult("llmrag", case["id"], "tutoring", True, "", level, outputs, list(case["prompts"]), [], retrieval_stats, score_tutoring_signals(case, outputs))
    except Exception as exc:
        return BenchmarkResult("llmrag", case["id"], "tutoring", False, str(exc), "", [], [], [], _empty_retrieval_stats(), {})


def _lexi_saved_stub(level: str) -> Dict[str, Any]:
    return {
        "display_name": "Benchmark User",
        "user_level": level,
        "diagnostic_results": {},
        "diagnostic_feedback": [],
        "learner_profile": {},
        "grammar_point_mastery": {},
        "level_source": "benchmark_seed",
        "level_confidence": "high",
    }


def run_lexi_tutoring(case: Dict[str, Any]) -> BenchmarkResult:
    learner_id = f"benchmark_{case['id']}"
    try:
        with project_context(ROOT / "LexiPath_German"):
            purge_modules()
            app_mod = importlib.import_module("app")
            learner_store = importlib.import_module("learner_store")

            learner_store.delete_learner(learner_id)
            state = app_mod.build_state_from_saved_learner(learner_id, _lexi_saved_stub(case["start_level"]))
            outputs: List[str] = []
            retrieval_stats = _empty_retrieval_stats()

            for prompt in case["prompts"]:
                state["messages"].append({"role": "user", "content": prompt})
                state = app_mod.app.invoke(state)
                outputs.append(state["messages"][-1]["content"])
                retrieval_stats["turns"] += 1
                docs = state.get("retrieved_documents", []) or []
                retrieval_stats["total_chunks"] += len(docs)
                if docs:
                    retrieval_stats["turns_with_chunks"] += 1
                if state.get("retrieval_used_fallback", False):
                    retrieval_stats["fallback_turns"] += 1

            learner_store.delete_learner(learner_id)
            return BenchmarkResult("Lexi_Path_German", case["id"], "tutoring", True, "", case["start_level"], outputs, list(case["prompts"]), [], retrieval_stats, score_tutoring_signals(case, outputs))
    except Exception as exc:
        return BenchmarkResult("Lexi_Path_German", case["id"], "tutoring", False, str(exc), "", [], [], [], _empty_retrieval_stats(), {})


def run_zeroshot_continuity(case: Dict[str, Any]) -> BenchmarkResult:
    level = case["gold_level"]
    try:
        with project_context(ROOT / "Zeroshot"):
            purge_modules()
            engine = importlib.import_module("engine")
            prompts = importlib.import_module("prompts")

            messages1 = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": f"Start teaching a learner at level {level}. Focus on their request and keep going naturally."},
            ]
            session1: List[str] = []
            first1 = engine.call_kimi(messages1)
            messages1.append({"role": "assistant", "content": first1})
            session1.append(first1)
            for user_msg in case["session1_messages"]:
                messages1.append({"role": "user", "content": user_msg})
                reply = engine.call_kimi(messages1)
                messages1.append({"role": "assistant", "content": reply})
                session1.append(reply)

            messages2 = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": "This is a new session. Continue from where we stopped yesterday."},
            ]
            session2: List[str] = []
            first2 = engine.call_kimi(messages2)
            messages2.append({"role": "assistant", "content": first2})
            session2.append(first2)
            for user_msg in case["session2_messages"]:
                messages2.append({"role": "user", "content": user_msg})
                reply = engine.call_kimi(messages2)
                messages2.append({"role": "assistant", "content": reply})
                session2.append(reply)

            return BenchmarkResult("Zeroshot", case["id"], "continuity", True, "", level, session1 + session2, list(case["session1_messages"]) + list(case["session2_messages"]), [], _empty_retrieval_stats(), score_continuity_signals(case, session1, session2))
    except Exception as exc:
        return BenchmarkResult("Zeroshot", case["id"], "continuity", False, str(exc), "", [], [], [], _empty_retrieval_stats(), {})


def run_llmrag_continuity(case: Dict[str, Any]) -> BenchmarkResult:
    level = case["gold_level"]
    try:
        with project_context(ROOT / "llmrag"):
            purge_modules()
            engine = importlib.import_module("engine")
            prompts = importlib.import_module("prompts")
            rag = importlib.import_module("rag")
            tutor = importlib.import_module("tutor")

            retrieval_stats = _empty_retrieval_stats()
            start_bundle = _llmrag_retrieve_bundle(level, "Start teaching me now with a suitable first lesson.", rag, tutor)
            retrieval_stats["turns"] += 1
            retrieval_stats["total_chunks"] += len(start_bundle["chunks"])
            if start_bundle["chunks"]:
                retrieval_stats["turns_with_chunks"] += 1
            if start_bundle["used_fallback"]:
                retrieval_stats["fallback_turns"] += 1

            messages1 = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": prompts.build_tutor_starter_prompt(level, start_bundle["context"])},
            ]
            session1: List[str] = []
            first1 = engine.call_chat(messages1)
            messages1.append({"role": "assistant", "content": first1})
            session1.append(first1)
            for user_msg in case["session1_messages"]:
                bundle = _llmrag_retrieve_bundle(level, user_msg, rag, tutor)
                retrieval_stats["turns"] += 1
                retrieval_stats["total_chunks"] += len(bundle["chunks"])
                if bundle["chunks"]:
                    retrieval_stats["turns_with_chunks"] += 1
                if bundle["used_fallback"]:
                    retrieval_stats["fallback_turns"] += 1
                messages1.append({"role": "user", "content": f"Retrieved context:\n{bundle['context']}\n\nLearner says: {user_msg}"})
                reply = engine.call_chat(messages1)
                messages1.append({"role": "assistant", "content": reply})
                session1.append(reply)

            bundle2 = _llmrag_retrieve_bundle(level, "This is a new session. Continue from where we stopped.", rag, tutor)
            retrieval_stats["turns"] += 1
            retrieval_stats["total_chunks"] += len(bundle2["chunks"])
            if bundle2["chunks"]:
                retrieval_stats["turns_with_chunks"] += 1
            if bundle2["used_fallback"]:
                retrieval_stats["fallback_turns"] += 1

            messages2 = [
                {"role": "system", "content": prompts.tutor_system_prompt(level)},
                {"role": "user", "content": "The learner is returning in a new session.\n\n" + prompts.build_tutor_starter_prompt(level, bundle2["context"])},
            ]
            session2: List[str] = []
            first2 = engine.call_chat(messages2)
            messages2.append({"role": "assistant", "content": first2})
            session2.append(first2)
            for user_msg in case["session2_messages"]:
                bundle = _llmrag_retrieve_bundle(level, user_msg, rag, tutor)
                retrieval_stats["turns"] += 1
                retrieval_stats["total_chunks"] += len(bundle["chunks"])
                if bundle["chunks"]:
                    retrieval_stats["turns_with_chunks"] += 1
                if bundle["used_fallback"]:
                    retrieval_stats["fallback_turns"] += 1
                messages2.append({"role": "user", "content": f"Retrieved context:\n{bundle['context']}\n\nLearner says: {user_msg}"})
                reply = engine.call_chat(messages2)
                messages2.append({"role": "assistant", "content": reply})
                session2.append(reply)

            return BenchmarkResult("llmrag", case["id"], "continuity", True, "", level, session1 + session2, list(case["session1_messages"]) + list(case["session2_messages"]), [], retrieval_stats, score_continuity_signals(case, session1, session2))
    except Exception as exc:
        return BenchmarkResult("llmrag", case["id"], "continuity", False, str(exc), "", [], [], [], _empty_retrieval_stats(), {})


def run_lexi_continuity(case: Dict[str, Any]) -> BenchmarkResult:
    learner_id = f"benchmark_{case['id']}"
    try:
        with project_context(ROOT / "LexiPath_German"):
            purge_modules()
            app_mod = importlib.import_module("app")
            learner_store = importlib.import_module("learner_store")

            learner_store.delete_learner(learner_id)
            state = app_mod.build_state_from_saved_learner(learner_id, _lexi_saved_stub(case["gold_level"]))
            session1: List[str] = []
            retrieval_stats = _empty_retrieval_stats()

            for user_msg in case["session1_messages"]:
                state["messages"].append({"role": "user", "content": user_msg})
                state = app_mod.app.invoke(state)
                session1.append(state["messages"][-1]["content"])
                retrieval_stats["turns"] += 1
                docs = state.get("retrieved_documents", []) or []
                retrieval_stats["total_chunks"] += len(docs)
                if docs:
                    retrieval_stats["turns_with_chunks"] += 1
                if state.get("retrieval_used_fallback", False):
                    retrieval_stats["fallback_turns"] += 1

            saved = learner_store.load_learner(learner_id) or {}
            state2 = app_mod.build_state_from_saved_learner(learner_id, saved)
            welcome_back = (
                f"Welcome back, {saved.get('display_name', 'Benchmark User')}. "
                f"I remember you around {saved.get('user_level', case['gold_level'])}. "
                "What would you like to work on today?"
            )
            state2["messages"].append({"role": "assistant", "content": welcome_back})
            session2: List[str] = [welcome_back]
            for user_msg in case["session2_messages"]:
                state2["messages"].append({"role": "user", "content": user_msg})
                state2 = app_mod.app.invoke(state2)
                session2.append(state2["messages"][-1]["content"])
                retrieval_stats["turns"] += 1
                docs = state2.get("retrieved_documents", []) or []
                retrieval_stats["total_chunks"] += len(docs)
                if docs:
                    retrieval_stats["turns_with_chunks"] += 1
                if state2.get("retrieval_used_fallback", False):
                    retrieval_stats["fallback_turns"] += 1

            learner_store.delete_learner(learner_id)
            return BenchmarkResult("Lexi_Path_German", case["id"], "continuity", True, "", case["gold_level"], session1 + session2, list(case["session1_messages"]) + list(case["session2_messages"]), [], retrieval_stats, score_continuity_signals(case, session1, session2))
    except Exception as exc:
        return BenchmarkResult("Lexi_Path_German", case["id"], "continuity", False, str(exc), "", [], [], [], _empty_retrieval_stats(), {})


RUNNERS = {
    ("diagnostic", "zeroshot"): run_zeroshot_diagnostic,
    ("diagnostic", "llmrag"): run_llmrag_diagnostic,
    ("diagnostic", "lexi"): run_lexi_diagnostic,
    ("tutoring", "zeroshot"): run_zeroshot_tutoring,
    ("tutoring", "llmrag"): run_llmrag_tutoring,
    ("tutoring", "lexi"): run_lexi_tutoring,
    ("continuity", "zeroshot"): run_zeroshot_continuity,
    ("continuity", "llmrag"): run_llmrag_continuity,
    ("continuity", "lexi"): run_lexi_continuity,
}

SYSTEM_NAME_MAP = {"zeroshot": "Zeroshot", "llmrag": "llmrag", "lexi": "Lexi_Path_German"}


def _result_to_json(result: BenchmarkResult) -> Dict[str, Any]:
    return {
        "system": result.system,
        "case_id": result.case_id,
        "track": result.track,
        "success": result.success,
        "error": result.error,
        "detected_level": result.detected_level,
        "assistant_outputs": result.assistant_outputs,
        "user_inputs": result.user_inputs,
        "diagnostic_trace": result.diagnostic_trace,
        "retrieval_stats": result.retrieval_stats,
        "signals": result.signals,
    }


def print_summary(results: List[BenchmarkResult]) -> None:
    print("\n=== Benchmark Summary ===")
    by_system: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        by_system.setdefault(result.system, []).append(result)

    for system, rows in by_system.items():
        successes = sum(1 for row in rows if row.success)
        print(f"- {system}: {successes}/{len(rows)} cases successful")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation for Zeroshot, llmrag, and Lexi_Path_German")
    parser.add_argument("--output", default="benchmark_report.json", help="Output JSON report path")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=["zeroshot", "llmrag", "lexi"],
        choices=["zeroshot", "llmrag", "lexi"],
        help="Subset of systems to benchmark",
    )
    parser.add_argument(
        "--tracks",
        nargs="*",
        default=["diagnostic", "tutoring", "continuity"],
        choices=["diagnostic", "tutoring", "continuity"],
        help="Subset of benchmark tracks to run",
    )
    args = parser.parse_args()

    selected_cases = [case for case in BENCHMARK_CASES if case["track"] in set(args.tracks)]

    started = time.time()
    results: List[BenchmarkResult] = []
    for case in selected_cases:
        for system in args.systems:
            print(f"Running {system} on {case['id']}...")
            runner = RUNNERS[(case["track"], system)]
            results.append(run_with_retries(runner, case))

    payload = {
        "generated_at_epoch": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
        "cases": selected_cases,
        "results": [_result_to_json(result) for result in results],
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nSaved benchmark report: {out_path}")


if __name__ == "__main__":
    main()
