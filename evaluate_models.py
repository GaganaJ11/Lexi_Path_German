import argparse
import importlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

DEFAULT_SCENARIO = {
    "name": "returning_learner_continuity",
    "gold_level": "B1",
    "learner_id": "eval_stateful_user",
    "display_name": "Eval User",
    "diagnostic_answers": [
        "Der Mann isst einen Apfel.",
        "Ich habe kein Auto.",
        "Ich wohne in Berlin.",
        "Ich habe gestern Deutsch gelernt.",
        "Ich lege das Buch auf den Tisch.",
        "Ein Auto ist schneller als ein Fahrrad.",
        "Ich lerne Deutsch, weil ich in Deutschland arbeiten will.",
        "Ich wuerde mehr reisen.",
        "Das ist die Frau, die mir hilft.",
    ],
    "session1_messages": [
        "I want to improve German for work communication.",
        "Please explain weil clauses and give one practice question.",
        "I find Perfekt a bit difficult.",
    ],
    "session2_messages": [
        "Can we continue from where we stopped yesterday?",
        "Give me a short exercise now.",
    ],
    "memory_keywords": ["weil", "perfekt", "work", "a2", "b1"],
}


@dataclass
class EvalResult:
    system_name: str
    success: bool
    error: str
    detected_level: str
    diagnostic_scores: Dict[int, int]
    session1_assistant: List[str]
    session2_assistant: List[str]
    session1_user: List[str]
    session2_user: List[str]
    retrieval_stats: Dict[str, Any]
    metrics: Dict[str, Any]


@contextmanager
def project_context(project_path: Path):
    prev_cwd = Path.cwd()
    prev_sys_path = list(sys.path)
    os.chdir(project_path)
    sys.path.insert(0, str(project_path))
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        sys.path = prev_sys_path


def purge_modules() -> None:
    for name in MODULES_TO_PURGE:
        if name in sys.modules:
            del sys.modules[name]


def _metrics_from_transcripts(session1: List[str], session2: List[str], memory_keywords: List[str]) -> Dict[str, Any]:
    all_replies = session1 + session2
    if not all_replies:
        return {
            "assistant_turns": 0,
            "avg_response_chars": 0,
            "followup_question_rate": 0.0,
            "session2_memory_keyword_hits": 0,
            "session2_memory_recall": False,
        }

    avg_len = sum(len(text) for text in all_replies) / len(all_replies)
    followups = sum(1 for text in all_replies if "?" in text)

    session2_blob = "\n".join(session2).lower()
    hits = sum(1 for kw in memory_keywords if kw.lower() in session2_blob)

    return {
        "assistant_turns": len(all_replies),
        "avg_response_chars": round(avg_len, 1),
        "followup_question_rate": round(followups / len(all_replies), 3),
        "session2_memory_keyword_hits": hits,
        "session2_memory_recall": hits > 0,
    }


def _empty_retrieval_stats() -> Dict[str, Any]:
    return {
        "turns": 0,
        "turns_with_chunks": 0,
        "total_chunks": 0,
        "fallback_turns": 0,
    }


def run_zeroshot(scenario: Dict[str, Any]) -> EvalResult:
    system = "Zeroshot"
    try:
        with project_context(ROOT / "Zeroshot"):
            purge_modules()
            diagnostic_logic = importlib.import_module("diagnostic_logic")
            engine = importlib.import_module("engine")
            prompts = importlib.import_module("prompts")
            syllabus = importlib.import_module("syllabus")
            tutor = importlib.import_module("tutor")

            results_by_task: Dict[int, int] = {}
            all_results: List[Dict[str, Any]] = []
            current_id = diagnostic_logic.DiagnosticManager.get_start_task_id()

            for answer in scenario["diagnostic_answers"]:
                if current_id is None:
                    break
                task = diagnostic_logic.DiagnosticManager.get_task(current_id)
                evaluation = diagnostic_logic.grade_diagnostic_answer(task, answer)

                all_results.append(
                    {
                        "id": task["id"],
                        "level": task["level"],
                        "topic": task["topic"],
                        "grammar_point": task["grammar_point"],
                        "score_value": evaluation["score_value"],
                    }
                )
                results_by_task[current_id] = evaluation["score_value"]
                current_id = diagnostic_logic.DiagnosticManager.get_next_task_id(current_id, results_by_task)

            detected_level = diagnostic_logic.DiagnosticManager.determine_final_level(results_by_task)
            summary = diagnostic_logic.build_summary(detected_level, all_results, results_by_task)

            formatted_syllabus = syllabus.format_syllabus_for_learner(detected_level)
            next_level = tutor._get_next_level(detected_level)
            starter_message = f"""
The learner has completed a diagnosis.

Detected level: {detected_level}
Strengths: {summary.get('strengths', [])}
Weaknesses: {summary.get('weaknesses', [])}

Structured syllabus:
{formatted_syllabus}

Learner preference:
focus on speaking

Please act as a warm, human-like private German tutor.
Important:
- explain the learner's level in a friendly way
- appreciate the learner after every answer
- follow the structured syllabus unless the learner asks otherwise
- after current level mastery, move to the next CEFR level automatically

In your first reply:
1. explain the learner's level in a warm way
2. encourage them
3. respond to their preference
4. start with the first module and first topic
5. ask one comfortable first exercise
6. mention that once {detected_level} is complete, continue with {next_level}
""".strip()

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": prompts.tutor_system_prompt(detected_level)},
                {"role": "user", "content": starter_message},
            ]

            session1_assistant: List[str] = []
            session2_assistant: List[str] = []

            first_reply = engine.call_ollama(messages)
            messages.append({"role": "assistant", "content": first_reply})
            session1_assistant.append(first_reply)

            for user_msg in scenario["session1_messages"]:
                messages.append({"role": "user", "content": user_msg})
                reply = engine.call_ollama(messages)
                messages.append({"role": "assistant", "content": reply})
                session1_assistant.append(reply)

            messages2: List[Dict[str, str]] = [
                {"role": "system", "content": prompts.tutor_system_prompt(detected_level)},
                {
                    "role": "user",
                    "content": "This is a new session. Start with a suitable lesson for my level.",
                },
            ]
            reply2_start = engine.call_ollama(messages2)
            messages2.append({"role": "assistant", "content": reply2_start})
            session2_assistant.append(reply2_start)

            for user_msg in scenario["session2_messages"]:
                messages2.append({"role": "user", "content": user_msg})
                reply = engine.call_ollama(messages2)
                messages2.append({"role": "assistant", "content": reply})
                session2_assistant.append(reply)

            metrics = _metrics_from_transcripts(session1_assistant, session2_assistant, scenario["memory_keywords"])
            return EvalResult(
                system_name=system,
                success=True,
                error="",
                detected_level=detected_level,
                diagnostic_scores=results_by_task,
                session1_assistant=session1_assistant,
                session2_assistant=session2_assistant,
                session1_user=list(scenario["session1_messages"]),
                session2_user=list(scenario["session2_messages"]),
                retrieval_stats=_empty_retrieval_stats(),
                metrics=metrics,
            )
    except Exception as exc:
        return EvalResult(system, False, str(exc), "", {}, [], [], [], [], _empty_retrieval_stats(), {})


def run_llmrag(scenario: Dict[str, Any]) -> EvalResult:
    system = "llmrag"
    try:
        with project_context(ROOT / "llmrag"):
            purge_modules()
            diagnostic_logic = importlib.import_module("diagnostic_logic")
            tutor = importlib.import_module("tutor")
            rag = importlib.import_module("rag")
            engine = importlib.import_module("engine")

            results: Dict[int, int] = {}
            current_id = diagnostic_logic.DiagnosticManager.get_start_task_id()

            for answer in scenario["diagnostic_answers"]:
                if current_id is None:
                    break
                task = diagnostic_logic.DiagnosticManager.get_task(current_id)
                grade = diagnostic_logic._grade_answer(task, answer)
                score = {"FULL": 2, "PARTIAL": 1, "FAIL": 0}.get(grade["label"], 0)
                results[current_id] = score
                current_id = diagnostic_logic.DiagnosticManager.get_next_task_id(current_id, score, results)

            detected_level = diagnostic_logic.DiagnosticManager.determine_final_level(results)

            def start_messages(first_user_message: str) -> List[Dict[str, str]]:
                query = tutor.build_retrieval_query(detected_level, first_user_message)
                chunks = rag.retrieve(query, level_filter=None)
                context = rag.format_context(chunks)
                return [
                    {"role": "system", "content": tutor.tutor_system_prompt(detected_level)},
                    {
                        "role": "user",
                        "content": (
                            f"Retrieved context:\n{context}\n\n"
                            f"Learner level: {detected_level}\n"
                            f"Learner request: {first_user_message}"
                        ),
                    },
                ]

            session1_assistant: List[str] = []
            session2_assistant: List[str] = []
            retrieval_stats = _empty_retrieval_stats()

            messages = start_messages("Start teaching me now with a suitable first lesson.")
            retrieval_stats["turns"] += 1
            starter_chunks = rag.retrieve(
                tutor.build_retrieval_query(detected_level, "Start teaching me now with a suitable first lesson."),
                level_filter=None,
            )
            retrieval_stats["total_chunks"] += len(starter_chunks)
            if starter_chunks:
                retrieval_stats["turns_with_chunks"] += 1
            first_reply = engine.call_chat(messages)
            messages.append({"role": "assistant", "content": first_reply})
            session1_assistant.append(first_reply)

            for user_msg in scenario["session1_messages"]:
                query = tutor.build_retrieval_query(detected_level, user_msg)
                chunks = rag.retrieve(query, level_filter=None)
                context = rag.format_context(chunks)
                retrieval_stats["turns"] += 1
                retrieval_stats["total_chunks"] += len(chunks)
                if chunks:
                    retrieval_stats["turns_with_chunks"] += 1
                messages.append({
                    "role": "user",
                    "content": f"Retrieved context:\n{context}\n\nLearner says: {user_msg}",
                })
                reply = engine.call_chat(messages)
                messages.append({"role": "assistant", "content": reply})
                session1_assistant.append(reply)

            messages2 = start_messages("This is a new session. Continue from my level.")
            retrieval_stats["turns"] += 1
            starter2_chunks = rag.retrieve(
                tutor.build_retrieval_query(detected_level, "This is a new session. Continue from my level."),
                level_filter=None,
            )
            retrieval_stats["total_chunks"] += len(starter2_chunks)
            if starter2_chunks:
                retrieval_stats["turns_with_chunks"] += 1
            reply2_start = engine.call_chat(messages2)
            messages2.append({"role": "assistant", "content": reply2_start})
            session2_assistant.append(reply2_start)

            for user_msg in scenario["session2_messages"]:
                query = tutor.build_retrieval_query(detected_level, user_msg)
                chunks = rag.retrieve(query, level_filter=None)
                context = rag.format_context(chunks)
                retrieval_stats["turns"] += 1
                retrieval_stats["total_chunks"] += len(chunks)
                if chunks:
                    retrieval_stats["turns_with_chunks"] += 1
                messages2.append({
                    "role": "user",
                    "content": f"Retrieved context:\n{context}\n\nLearner says: {user_msg}",
                })
                reply = engine.call_chat(messages2)
                messages2.append({"role": "assistant", "content": reply})
                session2_assistant.append(reply)

            metrics = _metrics_from_transcripts(session1_assistant, session2_assistant, scenario["memory_keywords"])
            return EvalResult(
                system_name=system,
                success=True,
                error="",
                detected_level=detected_level,
                diagnostic_scores=results,
                session1_assistant=session1_assistant,
                session2_assistant=session2_assistant,
                session1_user=list(scenario["session1_messages"]),
                session2_user=list(scenario["session2_messages"]),
                retrieval_stats=retrieval_stats,
                metrics=metrics,
            )
    except Exception as exc:
        return EvalResult(system, False, str(exc), "", {}, [], [], [], [], _empty_retrieval_stats(), {})


def run_lexipath_stateful(scenario: Dict[str, Any]) -> EvalResult:
    system = "Lexi_Path_German"
    try:
        with project_context(ROOT / "Lexi_Path_German"):
            purge_modules()
            app_mod = importlib.import_module("app")
            learner_store = importlib.import_module("learner_store")

            learner_id = scenario["learner_id"]
            display_name = scenario["display_name"]
            learner_store.delete_learner(learner_id)

            state = app_mod.build_initial_state(learner_id, display_name)
            state = app_mod.app.invoke(state)

            for answer in scenario["diagnostic_answers"]:
                if state.get("phase") == "tutoring":
                    break
                state["messages"].append({"role": "user", "content": answer})
                state = app_mod.app.invoke(state)

            detected_level = state.get("user_level", "")
            session1_assistant: List[str] = []
            session2_assistant: List[str] = []
            retrieval_stats = _empty_retrieval_stats()

            for user_msg in scenario["session1_messages"]:
                state["messages"].append({"role": "user", "content": user_msg})
                state = app_mod.app.invoke(state)
                session1_assistant.append(state["messages"][-1]["content"])
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
                f"Welcome back, {saved.get('display_name', display_name)}. "
                f"We'll continue from your current level, {saved.get('user_level', detected_level)}. "
                "What would you like to work on today?"
            )
            state2["messages"].append({"role": "assistant", "content": welcome_back})
            session2_assistant.append(welcome_back)

            for user_msg in scenario["session2_messages"]:
                state2["messages"].append({"role": "user", "content": user_msg})
                state2 = app_mod.app.invoke(state2)
                session2_assistant.append(state2["messages"][-1]["content"])
                retrieval_stats["turns"] += 1
                docs = state2.get("retrieved_documents", []) or []
                retrieval_stats["total_chunks"] += len(docs)
                if docs:
                    retrieval_stats["turns_with_chunks"] += 1
                if state2.get("retrieval_used_fallback", False):
                    retrieval_stats["fallback_turns"] += 1

            metrics = _metrics_from_transcripts(session1_assistant, session2_assistant, scenario["memory_keywords"])
            return EvalResult(
                system_name=system,
                success=True,
                error="",
                detected_level=detected_level,
                diagnostic_scores=state.get("diagnostic_results", {}),
                session1_assistant=session1_assistant,
                session2_assistant=session2_assistant,
                session1_user=list(scenario["session1_messages"]),
                session2_user=list(scenario["session2_messages"]),
                retrieval_stats=retrieval_stats,
                metrics=metrics,
            )
    except Exception as exc:
        return EvalResult(system, False, str(exc), "", {}, [], [], [], [], _empty_retrieval_stats(), {})


def _result_to_json(result: EvalResult) -> Dict[str, Any]:
    return {
        "system": result.system_name,
        "success": result.success,
        "error": result.error,
        "detected_level": result.detected_level,
        "diagnostic_scores": result.diagnostic_scores,
        "session1_user": result.session1_user,
        "session2_user": result.session2_user,
        "metrics": result.metrics,
        "retrieval_stats": result.retrieval_stats,
        "session1_assistant": result.session1_assistant,
        "session2_assistant": result.session2_assistant,
    }


def print_summary(results: List[EvalResult]) -> None:
    print("\n=== Evaluation Summary ===")
    for r in results:
        if not r.success:
            print(f"- {r.system_name}: FAILED ({r.error})")
            continue
        m = r.metrics
        print(
            f"- {r.system_name}: level={r.detected_level}, "
            f"turns={m.get('assistant_turns')}, "
            f"avg_chars={m.get('avg_response_chars')}, "
            f"followups={m.get('followup_question_rate')}, "
            f"session2_memory_recall={m.get('session2_memory_recall')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Zeroshot vs llmrag vs Lexi_Path_German")
    parser.add_argument("--output", default="evaluation_report.json", help="Output JSON report path")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=["zeroshot", "llmrag", "lexi"],
        choices=["zeroshot", "llmrag", "lexi"],
        help="Subset of systems to evaluate",
    )
    args = parser.parse_args()

    scenario = DEFAULT_SCENARIO

    runners = {
        "zeroshot": run_zeroshot,
        "llmrag": run_llmrag,
        "lexi": run_lexipath_stateful,
    }

    started = time.time()
    results: List[EvalResult] = []
    for key in args.systems:
        print(f"Running {key}...")
        results.append(runners[key](scenario))

    payload = {
        "scenario": scenario,
        "generated_at_epoch": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
        "results": [_result_to_json(r) for r in results],
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
