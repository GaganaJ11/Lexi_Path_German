import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
CURRICULUM_PATH = BASE_DIR / "nicos_curriculum_map.json"


GRAMMAR_POINT_TOPIC_HINTS = {
    "indefinite_articles_ein_eine_einen": "Articles",
    "definite_articles_basics": "Articles",
    "accusative_masculine_den": "Articles",
    "negation_kein": "Negation",
    "present_tense_basic_verbs": "Verb Conjugation",
    "perfect_tense_basics": "Verb Conjugation",
    "accusative_with_movement": "Cases",
    "comparatives_basics": "Grammar",
    "subordinate_clause_weil": "Sentence Structure",
    "relative_clauses_basics": "Sentence Structure",
    "konjunktiv_ii_basics": "Grammar",
}


@lru_cache(maxsize=1)
def load_curriculum() -> List[Dict[str, Any]]:
    if not CURRICULUM_PATH.exists():
        return []
    try:
        data = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _normalize_level(level: str) -> str:
    return level if level in {"A1", "A2", "B1"} else "A1"


def _lesson_matches_topic(lesson: Dict[str, Any], topic: str) -> bool:
    topics = lesson.get("primary_topics", [])
    if not isinstance(topics, list):
        return False
    return topic in topics


def _compact_lesson(lesson: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "mode": "guided",
        "lesson_id": lesson.get("lesson_id", ""),
        "level": lesson.get("level", ""),
        "title": lesson.get("title", ""),
        "primary_topics": lesson.get("primary_topics", []),
        "search_query": lesson.get("search_query", ""),
        "reason": reason,
    }


def select_syllabus_reference(
    level: str,
    topic: str,
    grammar_point: str = "",
    learner_profile: Dict[str, Any] = None,
) -> Dict[str, Any]:
    curriculum = load_curriculum()
    if not curriculum:
        return {"mode": "guided", "lesson_id": "", "reason": "No curriculum map available."}

    normalized_level = _normalize_level(level)
    topic_hint = GRAMMAR_POINT_TOPIC_HINTS.get(grammar_point, topic or "Grammar")
    profile = learner_profile or {}
    seen_lessons = set(profile.get("syllabus_history", []))

    level_lessons = [lesson for lesson in curriculum if lesson.get("level") == normalized_level]
    topic_matches = [lesson for lesson in level_lessons if _lesson_matches_topic(lesson, topic_hint)]
    unseen_topic_matches = [lesson for lesson in topic_matches if lesson.get("lesson_id") not in seen_lessons]

    if unseen_topic_matches:
        return _compact_lesson(unseen_topic_matches[0], f"Matched current {normalized_level} topic: {topic_hint}.")
    if topic_matches:
        return _compact_lesson(topic_matches[0], f"Reusing closest {normalized_level} topic reference: {topic_hint}.")
    if level_lessons:
        return _compact_lesson(level_lessons[0], f"Using first available {normalized_level} curriculum reference.")

    return _compact_lesson(curriculum[0], "Using first available curriculum reference.")


def format_syllabus_reference(reference: Dict[str, Any]) -> str:
    if not reference or not reference.get("lesson_id"):
        return "No syllabus reference is available for this turn."

    topics = reference.get("primary_topics", [])
    topic_text = ", ".join(topics) if isinstance(topics, list) and topics else "general German"
    return (
        f"Guided mode reference: {reference.get('level', 'A1')} "
        f"{reference.get('lesson_id', '')} - {reference.get('title', '')}. "
        f"Primary topics: {topic_text}. "
        f"Use this as a curriculum reference, not as a hard restriction. "
        f"Reason: {reference.get('reason', '')}"
    )
