import re
from typing import Dict

from engine import call_chat

DIAGNOSTIC_TASKS = [
    # A1
    {
        "id": 1,
        "level": "A1",
        "topic": "Articles",
        "grammar_point": "indefinite_articles_ein_eine_einen",
        "prompt_goal": "Check whether the learner can produce a simple accusative noun phrase with an indefinite article.",
        "criteria": "The answer should clearly show an accusative masculine phrase such as 'einen Apfel'.",
        "example_answer": "Der Mann isst einen Apfel.",
    },
    {
        "id": 2,
        "level": "A1",
        "topic": "Negation",
        "grammar_point": "negation_kein",
        "prompt_goal": "Check whether the learner can negate a noun phrase with 'kein'.",
        "criteria": "The answer should use 'kein' or an inflected form like 'keine' correctly.",
        "example_answer": "Nein, ich habe kein Auto.",
    },
    {
        "id": 3,
        "level": "A1",
        "topic": "Verb Conjugation",
        "grammar_point": "present_tense_basic_verbs",
        "prompt_goal": "Check whether the learner can write one simple present-tense sentence about themselves.",
        "criteria": "The answer should contain a clear present-tense sentence such as 'Ich wohne in Berlin.'",
        "example_answer": "Ich wohne in Berlin.",
    },

    # A2
    {
        "id": 4,
        "level": "A2",
        "topic": "Verb Conjugation",
        "grammar_point": "perfect_tense_basics",
        "prompt_goal": "Check whether the learner can describe a completed past action with Perfekt.",
        "criteria": "The answer should use a helper verb and a past participle appropriately.",
        "example_answer": "Ich habe gestern Deutsch gelernt.",
    },
    {
        "id": 5,
        "level": "A2",
        "topic": "Cases",
        "grammar_point": "accusative_with_movement",
        "prompt_goal": "Check whether the learner can use a two-way preposition with movement and accusative.",
        "criteria": "The answer should show movement toward a destination, such as 'auf den Tisch'.",
        "example_answer": "Ich lege das Buch auf den Tisch.",
    },
    {
        "id": 6,
        "level": "A2",
        "topic": "Grammar",
        "grammar_point": "comparatives_basics",
        "prompt_goal": "Check whether the learner can compare two things with a comparative and 'als'.",
        "criteria": "The answer should include a comparative form plus 'als'.",
        "example_answer": "Ein Auto ist schneller als ein Fahrrad.",
    },

    # B1
    {
        "id": 7,
        "level": "B1",
        "topic": "Sentence Structure",
        "grammar_point": "subordinate_clause_weil",
        "prompt_goal": "Check whether the learner can produce a 'weil' clause with the verb at the end.",
        "criteria": "The answer should contain a subordinate clause introduced by 'weil' with final verb placement.",
        "example_answer": "Ich lerne Deutsch, weil ich in Deutschland arbeiten will.",
    },
    {
        "id": 8,
        "level": "B1",
        "topic": "Grammar",
        "grammar_point": "konjunktiv_ii_basics",
        "prompt_goal": "Check whether the learner can express a hypothetical idea with Konjunktiv II.",
        "criteria": "The answer should use a form like 'wÃ¼rde' or another clear Konjunktiv II structure.",
        "example_answer": "Ich wÃ¼rde viel reisen und ein Haus kaufen.",
    },
    {
        "id": 9,
        "level": "B1",
        "topic": "Sentence Structure",
        "grammar_point": "relative_clauses_basics",
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


class DiagnosticManager:
    @staticmethod
    def get_start_task_id():
        return LEVEL_TASKS["A1"][0]

    @staticmethod
    def get_task(task_id):
        return next((task for task in DIAGNOSTIC_TASKS if task["id"] == task_id), None)

    @staticmethod
    def get_level_for_task(task_id):
        task = DiagnosticManager.get_task(task_id)
        return task["level"] if task else "A1"

    @staticmethod
    def get_topic_for_task(task_id):
        task = DiagnosticManager.get_task(task_id)
        return task["topic"] if task else "Grammar"

    @staticmethod
    def get_grammar_point_for_task(task_id):
        task = DiagnosticManager.get_task(task_id)
        return task["grammar_point"] if task else "general_grammar"

    @staticmethod
    def get_next_level(level):
        try:
            index = LEVEL_ORDER.index(level)
        except ValueError:
            return None
        next_index = index + 1
        if next_index >= len(LEVEL_ORDER):
            return None
        return LEVEL_ORDER[next_index]

    @staticmethod
    def get_level_results(level, results):
        return {
            task_id: score_value
            for task_id, score_value in results.items()
            if DiagnosticManager.get_level_for_task(task_id) == level
        }

    @staticmethod
    def count_points(level, results):
        return sum(DiagnosticManager.get_level_results(level, results).values())

    @staticmethod
    def count_failures(level, results):
        return sum(
            1
            for score_value in DiagnosticManager.get_level_results(level, results).values()
            if score_value == 0
        )

    @staticmethod
    def get_unasked_tasks(level, results):
        answered = set(
            task_id for task_id in results
            if DiagnosticManager.get_level_for_task(task_id) == level
        )
        return [task_id for task_id in LEVEL_TASKS[level] if task_id not in answered]

    @staticmethod
    def should_promote(level, results):
        return DiagnosticManager.count_points(level, results) >= PROMOTION_POINT_THRESHOLD[level]

    @staticmethod
    def should_stop_level(level, results):
        unasked = DiagnosticManager.get_unasked_tasks(level, results)
        failures = DiagnosticManager.count_failures(level, results)
        return failures >= FAIL_STOP_COUNT[level] or not unasked

    @staticmethod
    def get_next_task_id(current_id, score_value, results):
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
    def determine_final_level(results):
        if DiagnosticManager.count_points("B1", results) >= PROMOTION_POINT_THRESHOLD["B1"]:
            return "B1"
        if DiagnosticManager.count_points("A2", results) >= PROMOTION_POINT_THRESHOLD["A2"]:
            return "A2"
        return "A1"

    @staticmethod
    def score_by_level(results):
        return {
            level: DiagnosticManager.count_points(level, results)
            for level in LEVEL_ORDER
        }

    @staticmethod
    def grammar_point_scores(results):
        scores = {}
        for task_id, score_value in results.items():
            grammar_point = DiagnosticManager.get_grammar_point_for_task(task_id)
            if grammar_point not in scores:
                scores[grammar_point] = {"points": 0, "total": 0}
            scores[grammar_point]["points"] += score_value
            scores[grammar_point]["total"] += MAX_POINTS_PER_TASK
        return scores

    @staticmethod
    def format_question(task, generated_question):
        return (
            f"Level check for {task['level']} "
            f"({task['topic']} - {task['grammar_point']}):\n"
            f"{generated_question}"
        )


    @staticmethod
    def build_completion_message(final_level, results):
        scores = DiagnosticManager.score_by_level(results)
        return (
            f"Thanks for working through that with me. "
            f"Iâ€™d place you around {final_level} right now. "
            f"Your score summary is "
            f"A1={scores['A1']}/{MAX_POINTS_PER_LEVEL}, "
            f"A2={scores['A2']}/{MAX_POINTS_PER_LEVEL}, "
            f"B1={scores['B1']}/{MAX_POINTS_PER_LEVEL}. "
            f"From here, Iâ€™ll adjust my explanations so they feel manageable and useful for you."
        )



def _extract_field(text: str, field: str) -> str:
    pattern = rf"^{re.escape(field)}\s*:\s*(.+)$"
    for line in text.splitlines():
        match = re.match(pattern, line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _generate_question(task: Dict[str, str]) -> str:
    prompt = f"""
You are a warm German tutor writing one short placement question.

Target level: {task['level']}
Topic: {task['topic']}
Grammar point: {task['grammar_point']}
Goal: {task['prompt_goal']}

Rules:
- Ask for an answer in German.
- Use short, clear teacher wording.
- Keep it practical and natural.
- Do not include the answer.

Reply with only the question.
""".strip()

    try:
        content = call_chat([{"role": "user", "content": prompt}]).strip()
        return content or f"Please answer in German: {task['prompt_goal']}"
    except Exception:
        return f"Please answer in German: {task['prompt_goal']}"


def _fallback_grade(task: Dict[str, str], answer: str) -> Dict[str, str]:
    text = f" {answer.lower()} "
    gp = task.get("grammar_point", "")

    checks = {
        "indefinite_articles_ein_eine_einen": [" einen ", " eine ", " ein "],
        "negation_kein": [" kein ", " keine ", " keinen "],
        "present_tense_basic_verbs": [" ich ", " du ", " er ", " sie ", " wir "],
        "perfect_tense_basics": [" habe ", " hat ", " haben ", " bin ", " ist "],
        "accusative_with_movement": [" auf den ", " in den ", " an den "],
        "comparatives_basics": [" als ", "er "],
        "subordinate_clause_weil": [" weil "],
        "konjunktiv_ii_basics": [" wuerde ", " würde ", " haette ", " hätte "],
        "relative_clauses_basics": [", die ", ", der ", ", das "],
    }

    expected = checks.get(gp, [])
    hit_count = sum(1 for token in expected if token in text)

    if len(answer.strip()) < 3:
        return {"label": "FAIL", "rationale": "The answer is too short."}
    if expected and hit_count >= 1:
        return {"label": "PARTIAL", "rationale": "The target pattern appears partly."}
    if not expected and len(answer.split()) >= 4:
        return {"label": "PARTIAL", "rationale": "The answer is understandable but limited."}
    return {"label": "FAIL", "rationale": "The target grammar pattern is not clear."}


def _grade_answer(task: Dict[str, str], answer: str) -> Dict[str, str]:
    prompt = f"""
You are grading a short German placement answer.

Level: {task['level']}
Topic: {task['topic']}
Grammar point: {task['grammar_point']}
Goal: {task['prompt_goal']}
Criteria: {task['criteria']}
Reference answer: {task['example_answer']}
Learner answer: {answer}

Grade with one label:
- FULL
- PARTIAL
- FAIL

Reply exactly:
SCORE: FULL or PARTIAL or FAIL
RATIONALE: one short sentence
""".strip()

    try:
        raw = call_chat([{"role": "user", "content": prompt}]).strip()
        label = _extract_field(raw, "SCORE").upper()
        rationale = _extract_field(raw, "RATIONALE") or "Checked against the target grammar point."

        if label not in {"FULL", "PARTIAL", "FAIL"}:
            fallback = _fallback_grade(task, answer)
            label = fallback["label"]
            rationale = fallback["rationale"]

        return {"label": label, "rationale": rationale}
    except Exception:
        return _fallback_grade(task, answer)


def run_diagnosis() -> str:
    print("=" * 60)
    print("LexiPath-style diagnostic")
    print("Answer in German. Type 'exit' to quit.")
    print("=" * 60)

    results: Dict[int, int] = {}
    current_id = DiagnosticManager.get_start_task_id()

    while True:
        task = DiagnosticManager.get_task(current_id)
        question = _generate_question(task)
        prompt = DiagnosticManager.format_question(task, question)

        print(f"\nTutor: {prompt}")
        answer = input("You: ").strip()

        if answer.lower() in {"exit", "quit", "stop"}:
            print("Session ended.")
            return "A1"

        grade = _grade_answer(task, answer)
        score_map = {"FULL": 2, "PARTIAL": 1, "FAIL": 0}
        score = score_map.get(grade["label"], 0)
        results[current_id] = score

        print(f"Tutor: {grade['rationale']}")

        next_id = DiagnosticManager.get_next_task_id(current_id, score, results)
        if next_id is None:
            break
        current_id = next_id

    final_level = DiagnosticManager.determine_final_level(results)
    completion = DiagnosticManager.build_completion_message(final_level, results)

    print(f"\nTutor: {completion}")
    print(f"Detected level: {final_level}")
    return final_level


