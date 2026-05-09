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
        "question": 'Fill in the blank with the correct German article: "Ich sehe ___ Bruder." You can answer with just the missing word.',
    },
    {
        "id": 2,
        "level": "A1",
        "topic": "Negation",
        "grammar_point": "negation_kein",
        "prompt_goal": "Check whether the learner can negate a noun phrase with 'kein'.",
        "criteria": "The answer should use 'kein' or an inflected form like 'keine' correctly.",
        "example_answer": "Nein, ich habe kein Auto.",
        "question": 'Fill in the blank with the correct German negation: "Das ist ___ Apfel." You can answer with just the missing word.',
    },
    {
        "id": 3,
        "level": "A1",
        "topic": "Present tense",
        "grammar_point": "present_tense_basic_verbs",
        "prompt_goal": "Check whether the learner can write one simple present-tense sentence about themselves.",
        "criteria": "The answer should contain a clear present-tense sentence such as 'Ich wohne in Berlin.'",
        "example_answer": "Ich wohne in Berlin.",
        "question": 'Write one complete German sentence about something you do every day. Start with "Ich".',
    },

    # A2
    {
        "id": 4,
        "level": "A2",
        "topic": "Present tense",
        "grammar_point": "perfect_tense_basics",
        "prompt_goal": "Check whether the learner can describe a completed past action with Perfekt.",
        "criteria": "The answer should use a helper verb and a past participle appropriately.",
        "example_answer": "Ich habe gestern Deutsch gelernt.",
        "question": 'Write one German sentence about what you did yesterday. Use Perfekt, for example with "habe" or "bin".',
    },
    {
        "id": 5,
        "level": "A2",
        "topic": "Basic prepositions",
        "grammar_point": "accusative_with_movement",
        "prompt_goal": "Check whether the learner can use a two-way preposition with movement and accusative.",
        "criteria": "The answer should show movement toward a destination, such as 'auf den Tisch'.",
        "example_answer": "Ich lege das Buch auf den Tisch.",
        "question": 'Write one German sentence showing movement toward a place, using a phrase like "auf den", "in die", or "an das".',
    },
    {
        "id": 6,
        "level": "A2",
        "topic": "Sentence structure",
        "grammar_point": "comparatives_basics",
        "prompt_goal": "Check whether the learner can compare two things with a comparative and 'als'.",
        "criteria": "The answer should include a comparative form plus 'als'.",
        "example_answer": "Ein Auto ist schneller als ein Fahrrad.",
        "question": 'Write one German sentence comparing two things. Use a comparative form and "als".',
    },

    # B1
    {
        "id": 7,
        "level": "B1",
        "topic": "Sentence structure",
        "grammar_point": "subordinate_clause_weil",
        "prompt_goal": "Check whether the learner can produce a 'weil' clause with the verb at the end.",
        "criteria": "The answer should contain a subordinate clause introduced by 'weil' with final verb placement.",
        "example_answer": "Ich lerne Deutsch, weil ich in Deutschland arbeiten will.",
        "question": 'Answer in German: Why are you learning German? Use "weil" and put the verb at the end of the weil-clause.',
    },
    {
        "id": 8,
        "level": "B1",
        "topic": "Modal verbs",
        "grammar_point": "konjunktiv_ii_basics",
        "prompt_goal": "Check whether the learner can express a hypothetical idea with Konjunktiv II.",
        "criteria": "The answer should use a form like 'würde' or another clear Konjunktiv II structure.",
        "example_answer": "Ich würde viel reisen und ein Haus kaufen.",
        "question": 'Write one German sentence about what you would do if you had more free time. Use "würde", "hätte", or "wäre".',
    },
    {
        "id": 9,
        "level": "B1",
        "topic": "Sentence structure",
        "grammar_point": "relative_clauses_basics",
        "prompt_goal": "Check whether the learner can combine two clauses using a relative clause.",
        "criteria": "The answer should use a relative pronoun and a grammatically coherent relative clause.",
        "example_answer": "Das ist die Frau, die mir hilft.",
        "question": 'Combine this idea into one German sentence with a relative clause: "Das ist die Person. Die Person hilft mir."',
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
        return task["topic"] if task else "Sentence structure"

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
        remaining = DiagnosticManager.get_unasked_tasks(level, results)

        if remaining and not DiagnosticManager.should_stop_level(level, results):
            return remaining[0]

        if not remaining and DiagnosticManager.should_promote(level, results):
            next_level = DiagnosticManager.get_next_level(level)
            if next_level is None:
                return None
            next_level_tasks = DiagnosticManager.get_unasked_tasks(next_level, results)
            return next_level_tasks[0] if next_level_tasks else None

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
            f"({task['topic']}):\n"
            f"{generated_question}"
        )

    @staticmethod
    def build_completion_message(final_level, results):
        scores = DiagnosticManager.score_by_level(results)
        return (
            f"Thanks for working through that with me. "
            f"I’d place you around {final_level} right now. "
            f"Your score summary is "
            f"A1={scores['A1']}/{MAX_POINTS_PER_LEVEL}, "
            f"A2={scores['A2']}/{MAX_POINTS_PER_LEVEL}, "
            f"B1={scores['B1']}/{MAX_POINTS_PER_LEVEL}. "
            f"From here, I’ll adjust my explanations so they feel manageable and useful for you."
        )
