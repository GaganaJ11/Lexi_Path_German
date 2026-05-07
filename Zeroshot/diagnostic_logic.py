import math
import re
from typing import Dict, List, Optional, Tuple

from engine import call_kimi


LEVEL_ORDER = ["A1", "A2", "B1"]
MAX_POINTS_PER_TASK = 2

LEVEL_BLUEPRINTS = {
    "A1": [
        {
            "topic": "Articles",
            "grammar_point": "indefinite_articles_ein_eine_einen",
            "prompt_goal": "Check whether the learner can produce a simple accusative noun phrase with an indefinite article.",
            "criteria": "The answer should clearly show an accusative masculine phrase such as 'einen Apfel'.",
            "example_answer": "Der Mann isst einen Apfel.",
        },
        {
            "topic": "Negation",
            "grammar_point": "negation_kein",
            "prompt_goal": "Check whether the learner can negate a noun phrase with 'kein'.",
            "criteria": "The answer should use 'kein' or an inflected form like 'keine' correctly.",
            "example_answer": "Nein, ich habe kein Auto.",
        },
        {
            "topic": "Verb Conjugation",
            "grammar_point": "present_tense_basic_verbs",
            "prompt_goal": "Check whether the learner can write one simple present-tense sentence about themselves.",
            "criteria": "The answer should contain a clear present-tense sentence such as 'Ich wohne in Berlin.'",
            "example_answer": "Ich wohne in Berlin.",
        },
    ],
    "A2": [
        {
            "topic": "Verb Conjugation",
            "grammar_point": "perfect_tense_basics",
            "prompt_goal": "Check whether the learner can describe a completed past action with Perfekt.",
            "criteria": "The answer should use a helper verb and a past participle appropriately.",
            "example_answer": "Ich habe gestern Deutsch gelernt.",
        },
        {
            "topic": "Cases",
            "grammar_point": "accusative_with_movement",
            "prompt_goal": "Check whether the learner can use a two-way preposition with movement and accusative.",
            "criteria": "The answer should show movement toward a destination, such as 'auf den Tisch'.",
            "example_answer": "Ich lege das Buch auf den Tisch.",
        },
        {
            "topic": "Grammar",
            "grammar_point": "comparatives_basics",
            "prompt_goal": "Check whether the learner can compare two things with a comparative and 'als'.",
            "criteria": "The answer should include a comparative form plus 'als'.",
            "example_answer": "Ein Auto ist schneller als ein Fahrrad.",
        },
    ],
    "B1": [
        {
            "topic": "Sentence Structure",
            "grammar_point": "subordinate_clause_weil",
            "prompt_goal": "Check whether the learner can produce a 'weil' clause with the verb at the end.",
            "criteria": "The answer should contain a subordinate clause introduced by 'weil' with final verb placement.",
            "example_answer": "Ich lerne Deutsch, weil ich in Deutschland arbeiten will.",
        },
        {
            "topic": "Grammar",
            "grammar_point": "konjunktiv_ii_basics",
            "prompt_goal": "Check whether the learner can express a hypothetical idea with Konjunktiv II.",
            "criteria": "The answer should use a form like 'wuerde' or another clear Konjunktiv II structure.",
            "example_answer": "Ich wuerde viel reisen und ein Haus kaufen.",
        },
        {
            "topic": "Sentence Structure",
            "grammar_point": "relative_clauses_basics",
            "prompt_goal": "Check whether the learner can combine two clauses using a relative clause.",
            "criteria": "The answer should use a relative pronoun and a grammatically coherent relative clause.",
            "example_answer": "Das ist die Frau, die mir hilft.",
        },
    ],
}


def build_diagnostic_tasks() -> List[Dict]:
    tasks: List[Dict] = []
    next_id = 1

    for level in LEVEL_ORDER:
        for blueprint in LEVEL_BLUEPRINTS.get(level, []):
            task = {
                "id": next_id,
                "level": level,
                "topic": blueprint["topic"],
                "grammar_point": blueprint["grammar_point"],
                "prompt_goal": blueprint["prompt_goal"],
                "criteria": blueprint["criteria"],
                "example_answer": blueprint["example_answer"],
            }
            tasks.append(task)
            next_id += 1

    return tasks


DIAGNOSTIC_TASKS = build_diagnostic_tasks()


def extract_section(text: str, field_name: str) -> str:
    prefix = f"{field_name.upper()}:"
    for line in text.splitlines():
        if line.upper().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def call_model(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return call_kimi(messages).strip()


def build_fallback_diagnostic_question(task: Dict) -> str:
    question_map = {
        "indefinite_articles_ein_eine_einen": "Please write one short German sentence with a masculine accusative phrase, for example with 'einen ...'.",
        "negation_kein": "Please write one short German sentence that negates a noun with 'kein'.",
        "present_tense_basic_verbs": "Please write one simple present-tense sentence in German about yourself.",
        "perfect_tense_basics": "Please write one short German sentence about something you did yesterday.",
        "accusative_with_movement": "Please write one German sentence showing movement to a place, for example with 'auf den Tisch'.",
        "comparatives_basics": "Please compare two things in one German sentence using 'als'.",
        "subordinate_clause_weil": "Please write one German sentence with 'weil' to explain a reason.",
        "konjunktiv_ii_basics": "Please write one German sentence about a hypothetical situation, for example with 'wuerde'.",
        "relative_clauses_basics": "Please write one German sentence that includes a relative clause.",
    }
    return question_map.get(
        task["grammar_point"],
        f"Please answer in German with one short sentence about {task['topic'].lower()}.",
    )


def is_bad_diagnostic_question(text: str, task: Dict) -> bool:
    normalized = " ".join(text.lower().split())
    prompt_goal = " ".join(task["prompt_goal"].lower().split())

    banned_markers = [
        "check whether",
        "demonstrate",
        "target grammar point",
        "use correctly",
        "grammar point",
        "diagnostic goal",
        "learner can",
    ]

    if not normalized:
        return True
    if prompt_goal and (normalized == prompt_goal or prompt_goal in normalized):
        return True
    if any(marker in normalized for marker in banned_markers):
        return True
    if re.search(r"\b(can|should)\b.*\blearner\b", normalized):
        return True
    return False


class DiagnosticManager:
    @staticmethod
    def get_tasks_for_level(level: str) -> List[Dict]:
        return [task for task in DIAGNOSTIC_TASKS if task["level"] == level]

    @staticmethod
    def get_task_ids_for_level(level: str) -> List[int]:
        return [task["id"] for task in DiagnosticManager.get_tasks_for_level(level)]

    @staticmethod
    def get_start_task_id() -> Optional[int]:
        first_level = LEVEL_ORDER[0]
        task_ids = DiagnosticManager.get_task_ids_for_level(first_level)
        return task_ids[0] if task_ids else None

    @staticmethod
    def get_task(task_id: int) -> Optional[Dict]:
        return next((task for task in DIAGNOSTIC_TASKS if task["id"] == task_id), None)

    @staticmethod
    def get_level_for_task(task_id: int) -> str:
        task = DiagnosticManager.get_task(task_id)
        return task["level"] if task else LEVEL_ORDER[0]

    @staticmethod
    def get_grammar_point_for_task(task_id: int) -> str:
        task = DiagnosticManager.get_task(task_id)
        return task["grammar_point"] if task else "general_grammar"

    @staticmethod
    def get_topic_for_task(task_id: int) -> str:
        task = DiagnosticManager.get_task(task_id)
        return task["topic"] if task else "Grammar"

    @staticmethod
    def get_next_level(level: str) -> Optional[str]:
        try:
            index = LEVEL_ORDER.index(level)
        except ValueError:
            return None

        next_index = index + 1
        if next_index >= len(LEVEL_ORDER):
            return None
        return LEVEL_ORDER[next_index]

    @staticmethod
    def get_level_results(level: str, results: Dict[int, int]) -> Dict[int, int]:
        return {
            task_id: score_value
            for task_id, score_value in results.items()
            if DiagnosticManager.get_level_for_task(task_id) == level
        }

    @staticmethod
    def count_points(level: str, results: Dict[int, int]) -> int:
        return sum(DiagnosticManager.get_level_results(level, results).values())

    @staticmethod
    def count_failures(level: str, results: Dict[int, int]) -> int:
        return sum(
            1
            for score_value in DiagnosticManager.get_level_results(level, results).values()
            if score_value == 0
        )

    @staticmethod
    def get_unasked_tasks(level: str, results: Dict[int, int]) -> List[int]:
        answered = {
            task_id
            for task_id in results
            if DiagnosticManager.get_level_for_task(task_id) == level
        }
        return [
            task_id
            for task_id in DiagnosticManager.get_task_ids_for_level(level)
            if task_id not in answered
        ]

    @staticmethod
    def max_points_for_level(level: str) -> int:
        return len(DiagnosticManager.get_task_ids_for_level(level)) * MAX_POINTS_PER_TASK

    @staticmethod
    def promotion_points_for_level(level: str) -> int:
        max_points = DiagnosticManager.max_points_for_level(level)
        if max_points == 0:
            return 0
        return max(1, math.ceil(max_points * 0.67))

    @staticmethod
    def fail_stop_count_for_level(level: str) -> int:
        task_count = len(DiagnosticManager.get_task_ids_for_level(level))
        if task_count <= 1:
            return task_count
        return min(task_count, max(2, math.ceil(task_count * 0.5)))

    @staticmethod
    def should_promote(level: str, results: Dict[int, int]) -> bool:
        return DiagnosticManager.count_points(level, results) >= DiagnosticManager.promotion_points_for_level(level)

    @staticmethod
    def should_stop_level(level: str, results: Dict[int, int]) -> bool:
        unasked = DiagnosticManager.get_unasked_tasks(level, results)
        failures = DiagnosticManager.count_failures(level, results)
        fail_limit = DiagnosticManager.fail_stop_count_for_level(level)
        return failures >= fail_limit or not unasked

    @staticmethod
    def get_next_task_id(current_id: int, results: Dict[int, int]) -> Optional[int]:
        level = DiagnosticManager.get_level_for_task(current_id)

        if DiagnosticManager.should_promote(level, results):
            next_level = DiagnosticManager.get_next_level(level)
            if next_level is None:
                return None
            next_level_tasks = DiagnosticManager.get_unasked_tasks(next_level, results)
            return next_level_tasks[0] if next_level_tasks else None

        remaining = DiagnosticManager.get_unasked_tasks(level, results)
        if remaining and not DiagnosticManager.should_stop_level(level, results):
            return remaining[0]

        return None

    @staticmethod
    def determine_final_level(results: Dict[int, int]) -> str:
        final_level = LEVEL_ORDER[0]
        for level in LEVEL_ORDER:
            if DiagnosticManager.count_points(level, results) >= DiagnosticManager.promotion_points_for_level(level):
                final_level = level
        return final_level

    @staticmethod
    def score_by_level(results: Dict[int, int]) -> Dict[str, int]:
        return {
            level: DiagnosticManager.count_points(level, results)
            for level in LEVEL_ORDER
        }

    @staticmethod
    def grammar_point_scores(results: Dict[int, int]) -> Dict[str, Dict[str, int]]:
        scores: Dict[str, Dict[str, int]] = {}
        for task_id, score_value in results.items():
            grammar_point = DiagnosticManager.get_grammar_point_for_task(task_id)
            if grammar_point not in scores:
                scores[grammar_point] = {"points": 0, "total": 0}
            scores[grammar_point]["points"] += score_value
            scores[grammar_point]["total"] += MAX_POINTS_PER_TASK
        return scores

    @staticmethod
    def format_question(task: Dict, generated_question: str) -> str:
        return (
            f"Level check for {task['level']} ({task['topic']} - {task['grammar_point']}):\n"
            f"{generated_question}"
        )

    @staticmethod
    def build_completion_message(final_level: str, results: Dict[int, int]) -> str:
        scores = DiagnosticManager.score_by_level(results)
        parts = [
            f"{level}={scores[level]}/{DiagnosticManager.max_points_for_level(level)}"
            for level in LEVEL_ORDER
        ]
        return (
            f"Thanks for working through that with me. "
            f"I'd place you around {final_level} right now. "
            f"Your score summary is {', '.join(parts)}. "
            f"From here, I'll adjust my explanations so they feel manageable and useful for you."
        )


def print_intro() -> None:
    print("Welcome")
    print()
    print("Hi there! It's so nice to meet you. I'll ask just a few quick questions to get a sense of your current German level.")
    print("There's absolutely no pressure here—just share what feels comfortable, and answer in German whenever you can.")
    print()


def generate_diagnostic_question(task: Dict, user_level: str = "A1") -> str:
    system_prompt = (
        "You are Kimi, a warm German tutor creating one short level-check question. "
        "You use internal task notes to design a natural learner-facing prompt, not to copy their wording."
    )
    user_prompt = f"""
Target learner band: {task['level']}
Current learner estimate: {user_level}
Topic: {task['topic']}
Grammar point: {task['grammar_point']}
Diagnostic goal: {task['prompt_goal']}
Success criteria: {task['criteria']}
Reference answer: {task['example_answer']}

Write one short question or instruction that tests this grammar point.

Rules:
- The learner should answer in German.
- Keep the wording natural and teacher-like.
- Use English when giving the instruction.
- Treat the diagnostic goal, criteria, and reference answer as hidden design notes.
- Convert those notes into a concrete learner task.
- Do not include the answer.
- Keep it short.
- Do not label difficulty or mention CEFR.
- Do not repeat or paraphrase the diagnostic goal as the question.
- Turn the goal into a concrete learner task a teacher would actually ask.
- Ask for a sentence, comparison, question, or example the learner can produce directly.
- Avoid abstract wording like "demonstrate", "show", "use correctly", or "check whether".
- Do not copy phrases from the diagnostic goal unless they are essential grammar words like "weil" or "als".
- Make the learner's expected output obvious from the instruction itself.

Good output examples:
- Please write one short German sentence about something you did yesterday.
- Compare two things in one German sentence using "als".
- Write one German sentence with "weil" to explain a reason.

Bad output examples:
- Check whether you can describe a completed past action with Perfekt.
- Demonstrate the target grammar point in one sentence.

Reply with only the question text.
""".strip()

    try:
        generated = call_model(system_prompt, user_prompt)
        if generated:
            cleaned = generated.replace("\n", " ").strip()
            if not is_bad_diagnostic_question(cleaned, task):
                return cleaned
    except Exception:
        pass

    return build_fallback_diagnostic_question(task)


def grade_diagnostic_answer(task: Dict, user_answer: str) -> Dict:
    system_prompt = (
        "You are grading a German placement-test answer."
    )
    user_prompt = f"""
Level: {task['level']}
Topic: {task['topic']}
Grammar point: {task['grammar_point']}
Diagnostic goal: {task['prompt_goal']}
Criteria: {task['criteria']}
Reference answer: {task['example_answer']}
Student answer: {user_answer}

Grade with these labels:
- FULL: clearly demonstrates the target grammar point
- PARTIAL: partially demonstrates it, but with weaknesses or incompleteness
- FAIL: incorrect, avoids the target grammar, or is too weak to count

Be strict but fair.
Minor spelling mistakes are acceptable if the grammar target is still clear.

Reply exactly in this format:
SCORE: FULL or PARTIAL or FAIL
RATIONALE: one short sentence
""".strip()

    try:
        response = call_model(system_prompt, user_prompt)
        score_label = extract_section(response, "SCORE").upper()
        rationale = extract_section(response, "RATIONALE") or "I checked the answer against the target grammar."
    except Exception:
        score_label = "FAIL"
        rationale = "The automatic grading step did not return a usable answer."

    score_map = {
        "FULL": 2,
        "PARTIAL": 1,
        "FAIL": 0,
    }
    score_value = score_map.get(score_label, 0)

    return {
        "score_label": score_label if score_label in score_map else "FAIL",
        "score_value": score_value,
        "correct": score_value >= 1,
        "rationale": rationale,
    }


def build_human_feedback(task: Dict, evaluation: Dict, user_level: str = "A1") -> str:
    system_prompt = (
        "You are Kimi, a warm and supportive German tutor."
    )
    user_prompt = f"""
Student level estimate: {user_level}
Task topic: {task['topic']}
Grammar point: {task['grammar_point']}
Evaluation score: {evaluation['score_label']}
Evaluation rationale: {evaluation['rationale']}

Write a short tutor response after the learner answers a level-check question.

Rules:
- Sound human, warm, and supportive.
- Keep it short: 1 to 2 sentences.
- If FULL, acknowledge it naturally.
- If PARTIAL, be encouraging and signal that the learner is on the right track.
- If FAIL, be gentle and reassuring.
- Do not over-explain yet.
- Do not say diagnostic, verdict, yes, or no.

Reply with only the tutor message.
""".strip()

    try:
        reply = call_model(system_prompt, user_prompt)
        if reply:
            return reply
    except Exception:
        pass

    if evaluation["score_value"] == 2:
        return "Nice work. That was a strong answer."
    if evaluation["score_value"] == 1:
        return "Good start. You are on the right track."
    return "Good try. We will keep going one step at a time."


def describe_level(level: str) -> str:
    descriptions = {
        "A1": "beginner basics",
        "A2": "elementary German",
        "B1": "intermediate German",
    }
    return descriptions.get(level, level)


def build_study_plan(level: str) -> List[str]:
    plans = {
        "A1": [
            "Build confidence with self-introductions and everyday sentences.",
            "Practice articles, present tense verbs, and simple sentence order.",
            "Use short German answers in daily-life situations.",
        ],
        "A2": [
            "Practice Perfekt and connected everyday sentences.",
            "Improve control of cases, prepositions, and comparisons.",
            "Use German more confidently for work, travel, and routine communication.",
        ],
        "B1": [
            "Strengthen longer sentences with connectors and clause structure.",
            "Practice opinions, hypotheticals, and more natural written German.",
            "Build fluency for practical conversation and structured speaking.",
        ],
    }
    return plans[level]


def estimate_confidence(level: str, counts: Dict[str, Dict[str, int]]) -> float:
    level_count = len(DiagnosticManager.get_task_ids_for_level(level))
    if level_count == 0:
        return 0.7

    asked_ratio = counts[level]["answered"] / level_count
    point_ratio = counts[level]["points"] / DiagnosticManager.max_points_for_level(level)
    confidence = 0.65 + (0.2 * asked_ratio) + (0.15 * point_ratio)
    return round(min(0.95, confidence), 2)


def build_summary(level: str, all_results: List[Dict], score_map: Dict[int, int]) -> Dict:
    strengths: List[str] = []
    weaknesses: List[str] = []

    level_scores = DiagnosticManager.score_by_level(score_map)

    if level_scores["A1"] >= DiagnosticManager.promotion_points_for_level("A1"):
        strengths.append("You already have a solid command of beginner German basics.")
    else:
        weaknesses.append("You need more support with core beginner patterns like articles, negation, and basic sentence building.")

    if level_scores["A2"] >= DiagnosticManager.promotion_points_for_level("A2"):
        strengths.append("You can handle several everyday A2 grammar patterns with reasonable confidence.")
    elif any(item["level"] == "A2" for item in all_results):
        weaknesses.append("You need more practice with A2 structures such as Perfekt, movement prepositions, and comparisons.")

    if level_scores["B1"] >= DiagnosticManager.promotion_points_for_level("B1"):
        strengths.append("You show good potential with longer intermediate structures and more flexible expression.")
    elif any(item["level"] == "B1" for item in all_results):
        weaknesses.append("You still need more work on B1 sentence structure, hypotheticals, and clause linking.")

    counts: Dict[str, Dict[str, int]] = {}
    for current_level in LEVEL_ORDER:
        answered = [item for item in all_results if item["level"] == current_level]
        counts[current_level] = {
            "answered": len(answered),
            "points": sum(item["score_value"] for item in answered),
            "max_points": DiagnosticManager.max_points_for_level(current_level),
        }

    return {
        "detected_level": level,
        "confidence": estimate_confidence(level, counts),
        "counts": counts,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "study_plan": build_study_plan(level),
    }


def print_transition_message(message: str) -> None:
    print("\nTutor:")
    print(message)
    print()


def run_diagnostic() -> Tuple[List[Dict], Dict]:
    print_intro()

    results_by_task: Dict[int, int] = {}
    all_results: List[Dict] = []
    current_id = DiagnosticManager.get_start_task_id()

    if current_id is None:
        summary = build_summary(LEVEL_ORDER[0], [], {})
        return [], summary

    current_level = DiagnosticManager.get_level_for_task(current_id)

    print_transition_message(
        f"We will begin with a few questions around {describe_level(current_level)}."
    )

    while current_id is not None:
        task = DiagnosticManager.get_task(current_id)
        if not task:
            break

        generated_question = generate_diagnostic_question(task, user_level=current_level)

        print("\nTutor:")
        print(DiagnosticManager.format_question(task, generated_question))
        print()

        user_answer = input("You: ").strip()
        evaluation = grade_diagnostic_answer(task, user_answer)
        feedback = build_human_feedback(task, evaluation, user_level=current_level)

        print("\nTutor:")
        print(feedback)
        print()

        all_results.append(
            {
                "id": task["id"],
                "level": task["level"],
                "topic": task["topic"],
                "grammar_point": task["grammar_point"],
                "question": generated_question,
                "user_answer": user_answer,
                "score_label": evaluation["score_label"],
                "score_value": evaluation["score_value"],
                "is_correct": evaluation["correct"],
                "explanation": evaluation["rationale"],
            }
        )

        results_by_task[current_id] = evaluation["score_value"]
        next_id = DiagnosticManager.get_next_task_id(current_id, results_by_task)

        if next_id is not None:
            next_level = DiagnosticManager.get_level_for_task(next_id)
            if next_level != current_level:
                print_transition_message(
                    f"Nice progress. Let us check a little {describe_level(next_level)} next."
                )
                current_level = next_level
        current_id = next_id

    final_level = DiagnosticManager.determine_final_level(results_by_task)
    summary = build_summary(final_level, all_results, results_by_task)

    print(DiagnosticManager.build_completion_message(final_level, results_by_task))
    print()

    return all_results, summary
