from typing import Dict, List, Tuple

from engine import call_ollama


DIAGNOSTIC_TASKS = [
    {
        "id": 1,
        "level": "A1",
        "topic": "Articles",
        "grammar_point": "indefinite_articles_ein_eine_einen",
        "question_text": "Please write one short German sentence with a masculine accusative noun phrase using an indefinite article, for example something like '... einen ...'.",
        "prompt_goal": "Check whether the learner can produce a simple accusative noun phrase with an indefinite article.",
        "criteria": "The answer should clearly show an accusative masculine phrase such as 'einen Apfel'.",
        "example_answer": "Der Mann isst einen Apfel.",
    },
    {
        "id": 2,
        "level": "A1",
        "topic": "Negation",
        "grammar_point": "negation_kein",
        "question_text": "Please answer in one short German sentence and negate a noun with 'kein'.",
        "prompt_goal": "Check whether the learner can negate a noun phrase with 'kein'.",
        "criteria": "The answer should use 'kein' or an inflected form like 'keine' correctly.",
        "example_answer": "Nein, ich habe kein Auto.",
    },
    {
        "id": 3,
        "level": "A1",
        "topic": "Verb Conjugation",
        "grammar_point": "present_tense_basic_verbs",
        "question_text": "Please write one simple present-tense sentence in German about yourself.",
        "prompt_goal": "Check whether the learner can write one simple present-tense sentence about themselves.",
        "criteria": "The answer should contain a clear present-tense sentence such as 'Ich wohne in Berlin.'",
        "example_answer": "Ich wohne in Berlin.",
    },
    {
        "id": 4,
        "level": "A2",
        "topic": "Verb Conjugation",
        "grammar_point": "perfect_tense_basics",
        "question_text": "Please write one short German sentence in the Perfekt tense about something you did yesterday.",
        "prompt_goal": "Check whether the learner can describe a completed past action with Perfekt.",
        "criteria": "The answer should use a helper verb and a past participle appropriately.",
        "example_answer": "Ich habe gestern Deutsch gelernt.",
    },
    {
        "id": 5,
        "level": "A2",
        "topic": "Cases",
        "grammar_point": "accusative_with_movement",
        "question_text": "Please write one German sentence that shows movement to a destination with a two-way preposition and accusative, for example 'auf den Tisch'.",
        "prompt_goal": "Check whether the learner can use a two-way preposition with movement and accusative.",
        "criteria": "The answer should show movement toward a destination, such as 'auf den Tisch'.",
        "example_answer": "Ich lege das Buch auf den Tisch.",
    },
    {
        "id": 6,
        "level": "A2",
        "topic": "Grammar",
        "grammar_point": "comparatives_basics",
        "question_text": "Please compare two things in one German sentence using a comparative and 'als'.",
        "prompt_goal": "Check whether the learner can compare two things with a comparative and 'als'.",
        "criteria": "The answer should include a comparative form plus 'als'.",
        "example_answer": "Ein Auto ist schneller als ein Fahrrad.",
    },
    {
        "id": 7,
        "level": "B1",
        "topic": "Sentence Structure",
        "grammar_point": "subordinate_clause_weil",
        "question_text": "Please write one German sentence with 'weil' and put the verb at the end of the subordinate clause.",
        "prompt_goal": "Check whether the learner can produce a 'weil' clause with the verb at the end.",
        "criteria": "The answer should contain a subordinate clause introduced by 'weil' with final verb placement.",
        "example_answer": "Ich lerne Deutsch, weil ich in Deutschland arbeiten will.",
    },
    {
        "id": 8,
        "level": "B1",
        "topic": "Grammar",
        "grammar_point": "konjunktiv_ii_basics",
        "question_text": "Please write one German sentence about a hypothetical situation using Konjunktiv II, for example with 'wuerde'.",
        "prompt_goal": "Check whether the learner can express a hypothetical idea with Konjunktiv II.",
        "criteria": "The answer should use a form like 'wuerde' or another clear Konjunktiv II structure.",
        "example_answer": "Ich wuerde viel reisen und ein Haus kaufen.",
    },
    {
        "id": 9,
        "level": "B1",
        "topic": "Sentence Structure",
        "grammar_point": "relative_clauses_basics",
        "question_text": "Please write one German sentence that includes a relative clause.",
        "prompt_goal": "Check whether the learner can combine two clauses using a relative clause.",
        "criteria": "The answer should use a relative pronoun and a grammatically coherent relative clause.",
        "example_answer": "Das ist die Frau, die mir hilft.",
    },
]

LEVEL_ORDER = ["A1", "A2", "B1"]

LEVEL_TASKS = {
    "A1": [1, 2, 3],
    "A2": [4, 5, 6],
    "B1": [7, 8, 9],
}

PROMOTION_POINT_THRESHOLD = {
    "A1": 4,
    "A2": 4,
    "B1": 4,
}

FAIL_STOP_COUNT = {
    "A1": 2,
    "A2": 2,
    "B1": 2,
}

MAX_POINTS_PER_TASK = 2
MAX_POINTS_PER_LEVEL = 6


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
    return call_ollama(messages).strip()


class DiagnosticManager:
    @staticmethod
    def get_start_task_id() -> int:
        return LEVEL_TASKS["A1"][0]

    @staticmethod
    def get_task(task_id: int) -> Dict:
        return next((task for task in DIAGNOSTIC_TASKS if task["id"] == task_id), None)

    @staticmethod
    def get_level_for_task(task_id: int) -> str:
        task = DiagnosticManager.get_task(task_id)
        return task["level"] if task else "A1"

    @staticmethod
    def get_grammar_point_for_task(task_id: int) -> str:
        task = DiagnosticManager.get_task(task_id)
        return task["grammar_point"] if task else "general_grammar"

    @staticmethod
    def get_next_level(level: str) -> str:
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
        return [task_id for task_id in LEVEL_TASKS[level] if task_id not in answered]

    @staticmethod
    def should_promote(level: str, results: Dict[int, int]) -> bool:
        return DiagnosticManager.count_points(level, results) >= PROMOTION_POINT_THRESHOLD[level]

    @staticmethod
    def should_stop_level(level: str, results: Dict[int, int]) -> bool:
        unasked = DiagnosticManager.get_unasked_tasks(level, results)
        failures = DiagnosticManager.count_failures(level, results)
        return failures >= FAIL_STOP_COUNT[level] or not unasked

    @staticmethod
    def get_next_task_id(current_id: int, results: Dict[int, int]) -> int:
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
        if DiagnosticManager.count_points("B1", results) >= PROMOTION_POINT_THRESHOLD["B1"]:
            return "B1"
        if DiagnosticManager.count_points("A2", results) >= PROMOTION_POINT_THRESHOLD["A2"]:
            return "A2"
        return "A1"

    @staticmethod
    def score_by_level(results: Dict[int, int]) -> Dict[str, int]:
        return {
            level: DiagnosticManager.count_points(level, results)
            for level in LEVEL_ORDER
        }

    @staticmethod
    def format_question(task: Dict, generated_question: str) -> str:
        return (
            f"Level check for {task['level']} ({task['topic']} - {task['grammar_point']}):\n"
            f"{generated_question}"
        )

    @staticmethod
    def get_question_text(task_id: int) -> str:
        task = DiagnosticManager.get_task(task_id)
        if not task:
            return "Please answer in German."
        return task.get("question_text") or f"Please answer in German: {task['prompt_goal']}"

    @staticmethod
    def build_completion_message(final_level: str, results: Dict[int, int]) -> str:
        scores = DiagnosticManager.score_by_level(results)
        return (
            f"Thanks for working through that with me. "
            f"I would place you around {final_level} right now. "
            f"Your score summary is "
            f"A1={scores['A1']}/{MAX_POINTS_PER_LEVEL}, "
            f"A2={scores['A2']}/{MAX_POINTS_PER_LEVEL}, "
            f"B1={scores['B1']}/{MAX_POINTS_PER_LEVEL}."
        )


def print_intro() -> None:
    print("=" * 60)
    print("German Placement Check")
    print("=" * 60)
    print("I will ask a few short questions to estimate your German level.")
    print("Please answer in German as naturally as you can.")
    print("If you are unsure, just try your best.")
    print()


def generate_diagnostic_question(task: Dict, user_level: str = "A1") -> str:
    return task.get("question_text") or f"Please answer in German: {task['prompt_goal']}"


def grade_diagnostic_answer(task: Dict, user_answer: str) -> Dict:
    system_prompt = (
        "You are grading a German placement-test answer. "
        "Be strict but fair and follow the requested output format exactly."
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
        "You are a warm and supportive German tutor. "
        "Write a short feedback message only."
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


def build_summary(level: str, all_results: List[Dict], score_map: Dict[int, int]) -> Dict:
    strengths: List[str] = []
    weaknesses: List[str] = []

    level_scores = DiagnosticManager.score_by_level(score_map)

    if level_scores["A1"] >= 4:
        strengths.append("You already have a solid command of beginner German basics.")
    else:
        weaknesses.append("You need more support with core beginner patterns like articles, negation, and basic sentence building.")

    if level_scores["A2"] >= 4:
        strengths.append("You can handle several everyday A2 grammar patterns with reasonable confidence.")
    elif any(item["level"] == "A2" for item in all_results):
        weaknesses.append("You need more practice with A2 structures such as Perfekt, movement prepositions, and comparisons.")

    if level_scores["B1"] >= 4:
        strengths.append("You show good potential with longer intermediate structures and more flexible expression.")
    elif any(item["level"] == "B1" for item in all_results):
        weaknesses.append("You still need more work on B1 sentence structure, hypotheticals, and clause linking.")

    counts = {}
    for current_level in LEVEL_ORDER:
        answered = [item for item in all_results if item["level"] == current_level]
        counts[current_level] = {
            "answered": len(answered),
            "points": sum(item["score_value"] for item in answered),
            "max_points": len(answered) * MAX_POINTS_PER_TASK,
        }

    confidence_map = {
        "A1": 0.84,
        "A2": 0.80,
        "B1": 0.76,
    }

    return {
        "detected_level": level,
        "confidence": confidence_map[level],
        "counts": counts,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "study_plan": build_study_plan(level),
    }


def print_transition_message(message: str) -> None:
    print("-" * 60)
    print(message)
    print("-" * 60)
    print()


def run_diagnostic() -> Tuple[List[Dict], Dict]:
    print_intro()

    results_by_task: Dict[int, int] = {}
    all_results: List[Dict] = []
    current_id = DiagnosticManager.get_start_task_id()
    current_level = DiagnosticManager.get_level_for_task(current_id)

    print_transition_message(
        f"We will begin with a few questions around {describe_level(current_level)}."
    )

    while current_id is not None:
        task = DiagnosticManager.get_task(current_id)
        generated_question = generate_diagnostic_question(task, user_level=current_level)

        print("=" * 60)
        print(DiagnosticManager.format_question(task, generated_question))
        print()

        user_answer = input("Your answer: ").strip()
        evaluation = grade_diagnostic_answer(task, user_answer)
        feedback = build_human_feedback(task, evaluation, user_level=current_level)

        print(feedback)
        print(f"Quick note: {evaluation['rationale']}")
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
