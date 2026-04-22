GERMAN_SYLLABUS = {
    "A1": {
        "title": "A1 – Beginner (Foundation Level)",
        "goal": "Understand and use basic everyday German.",

        "communication_topics": [
            "Greetings and introductions",
            "Personal information",
            "Numbers and alphabet",
            "Family and people",
            "Food and shopping",
            "Time, days, and daily routine",
            "Hobbies and activities",
            "Talking about your home and city",
            "Simple conversations in Germany"
        ],

        "grammar_topics": [
            "Simple sentence structure",
            "Verb conjugation: sein, haben, regular verbs",
            "Articles: der, die, das",
            "Personal pronouns",
            "Question forms",
            "Accusative basics",
            "Negation: nicht, kein",
            "Plural basics",
            "Modal verbs basics"
        ],

        "real_life_practice": [
            "Introducing yourself",
            "Ordering food",
            "Asking and answering simple questions",
            "Talking about your daily life",
            "Shopping and prices",
            "Basic doctor/pharmacy situations"
        ]
    },

    "A2": {
        "title": "A2 – Elementary (Communication Level)",
        "goal": "Handle daily situations and simple conversations more confidently.",

        "communication_topics": [
            "Talking about past events",
            "Work and jobs",
            "Travel and directions",
            "Health and lifestyle",
            "Messages, calls, and appointments",
            "Making plans",
            "Giving opinions simply",
            "Asking for help",
            "Describing experiences"
        ],

        "grammar_topics": [
            "Perfekt tense",
            "Dative and accusative together",
            "Modal verbs in more detail",
            "Subordinate clauses: weil, dass, wenn",
            "Reflexive verbs",
            "Comparisons",
            "Prepositions of place and time",
            "Word order in longer sentences",
            "Separable verbs"
        ],

        "real_life_practice": [
            "Talking about yesterday",
            "Writing short messages",
            "Making appointments",
            "Using transport and asking for directions",
            "Talking about work and routines",
            "Describing problems and asking for support"
        ]
    },

    "B1": {
        "title": "B1 – Intermediate (Fluency Level)",
        "goal": "Speak more independently, clearly, and with better structure.",

        "communication_topics": [
            "Expressing opinions",
            "Giving reasons and explanations",
            "Talking about goals and plans",
            "Work and career",
            "Relationships and emotions",
            "Media and technology",
            "Culture and society",
            "Discussions and arguments",
            "Formal and semi-formal communication"
        ],

        "grammar_topics": [
            "Perfekt and Präteritum",
            "Introduction to Plusquamperfekt",
            "Passive voice",
            "Konjunktiv II",
            "Relative clauses",
            "Connectors: obwohl, nachdem, während, deshalb",
            "Infinitive structures",
            "Adjective endings basics",
            "More advanced word order"
        ],

        "real_life_practice": [
            "Job interviews",
            "Complaints and formal situations",
            "Writing structured texts",
            "Giving opinions in discussion",
            "Talking about experiences and future plans",
            "Handling everyday problems in German"
        ]
    }
}


LEVEL_ORDER = ["A1", "A2", "B1"]


def get_learning_path_from_level(level: str) -> list:
    if level == "Pre-A1":
        start_idx = 0
    else:
        start_idx = LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 0
    return LEVEL_ORDER[start_idx:]


def format_syllabus_for_learner(level: str) -> str:
    path = get_learning_path_from_level(level)

    lines = []
    lines.append("Your German Learning Path")
    lines.append("=" * 30)
    lines.append("Based on your current level, here is a structured plan we can follow.")
    lines.append("You can skip topics you already know, ask for more examples, or focus on a specific skill.")
    lines.append("")

    for lvl in path:
        data = GERMAN_SYLLABUS[lvl]
        lines.append(data["title"])
        lines.append(f"Goal: {data['goal']}")
        lines.append("")

        lines.append("Communication Topics:")
        for topic in data["communication_topics"]:
            lines.append(f" - {topic}")
        lines.append("")

        lines.append("Grammar Topics:")
        for topic in data["grammar_topics"]:
            lines.append(f" - {topic}")
        lines.append("")

        lines.append("Real-Life Practice:")
        for topic in data["real_life_practice"]:
            lines.append(f" - {topic}")
        lines.append("")
        lines.append("-" * 30)

    lines.append("")
    lines.append("How we will learn:")
    lines.append("1. Learn one small topic")
    lines.append("2. See examples")
    lines.append("3. You try")
    lines.append("4. I correct and explain gently")
    lines.append("5. Then we decide whether to practice more or move ahead")

    return "\n".join(lines)