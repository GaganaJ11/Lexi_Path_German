import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from config import COLLECTION_NAME, EMBEDDING_MODEL, get_database_url

TRUSTED_SOURCES = (
    "LexiPath_ManualRules",
    "Nicos-Weg-GitHub",
    "DiscoResearch",
    "Avemio_ReasoningDE",
)

TOPIC_KEYWORDS = {
    "Articles": ["article", "artikel", "der", "die", "das", "ein", "eine", "einen", "dem", "den"],
    "Negation": ["negation", "negative", "kein", "keine", "nicht", "verneinung"],
    "Verb Conjugation": ["verb", "conjugation", "konjugation", "perfekt", "partizip", "past tense"],
    "Sentence Structure": ["word order", "sentence structure", "stellung", "weil", "nebensatz", "relative clause"],
    "Cases": ["case", "akkusativ", "dativ", "nominativ", "genitiv", "preposition", "praeposition"],
    "Grammar": ["grammar", "grammatik", "konjunktiv", "comparative", "vergleich"],
}

GRAMMAR_POINT_KEYWORDS = {
    "accusative_masculine_den": [" den ", "accusative", "akkusativ", "direct object"],
    "definite_articles_basics": ["article", "artikel", "der", "die", "das"],
    "indefinite_articles_ein_eine_einen": [" ein ", " eine ", " einen "],
    "negation_kein": [" kein ", " keine ", " negation ", " verneinung ", " not any "],
    "present_tense_basic_verbs": ["verb", "present", "conjugation", "konjugation"],
    "perfect_tense_basics": ["perfekt", "past", "partizip", "haben", "sein"],
    "accusative_with_movement": [" auf den ", "movement", "akkusativ"],
    "comparatives_basics": ["comparative", "komparativ", " als ", "faster", "schneller"],
    "subordinate_clause_weil": [" weil ", "nebensatz", "subordinate", "verb at the end"],
    "konjunktiv_ii_basics": [" würde ", "wuerde", "konjunktiv", "hypothetical", "would"],
    "relative_clauses_basics": ["relative clause", "relativsatz", " die mir ", " der ", " die ", " das "],
}

TOPIC_DEFAULT_GRAMMAR_POINTS = {
    "Articles": "definite_articles_basics",
    "Negation": "negation_kein",
    "Verb Conjugation": "present_tense_basic_verbs",
    "Sentence Structure": "subordinate_clause_weil",
    "Cases": "accusative_with_movement",
    "Grammar": "general_grammar",
}

LEVEL_FALLBACKS = {
    "B1": ["B1", "A2", "A1"],
    "A2": ["A2", "A1"],
    "A1": ["A1"],
}


@lru_cache(maxsize=1)
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=get_database_url(),
        use_jsonb=True,
    )


def normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return f" {normalized} "


def infer_topic(query: str) -> str:
    lowered = query.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "Grammar"


def infer_grammar_point(query: str, topic: str) -> str:
    normalized_query = normalize_text(query)
    alpha_query = normalize_text(re.sub(r"[^a-zA-ZäöüÄÖÜß ]+", " ", query))

    for grammar_point, keywords in GRAMMAR_POINT_KEYWORDS.items():
        if any(keyword in normalized_query or keyword in alpha_query for keyword in keywords):
            return grammar_point

    return TOPIC_DEFAULT_GRAMMAR_POINTS.get(topic, "general_grammar")


def search_documents(query: str, metadata_filter: Dict[str, str], k: int):
    try:
        return get_vector_store().similarity_search(
            query,
            k=k,
            filter=metadata_filter,
        )
    except Exception:
        return []


def deduplicate_documents(documents):
    unique = []
    seen = set()
    for doc in documents:
        if doc.metadata.get("source") not in TRUSTED_SOURCES:
            continue
        stable_metadata = {
            key: value for key, value in doc.metadata.items()
            if key != "learner_state_score"
        }
        key = (doc.page_content, tuple(sorted(stable_metadata.items())))
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _as_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _normalize_signal(value: str) -> str:
    return normalize_text(str(value).replace("_", " "))


def _signal_matches_document(signal: str, doc) -> bool:
    normalized_signal = _normalize_signal(signal)
    metadata = doc.metadata
    searchable = " ".join(
        [
            str(metadata.get("level", "")),
            str(metadata.get("topic", "")),
            str(metadata.get("grammar_point", "")).replace("_", " "),
            str(metadata.get("lesson_id", "")),
            doc.page_content[:240],
        ]
    )
    return normalized_signal.strip() in _normalize_signal(searchable)


def _current_goal_keywords(learner_profile: Dict[str, Any]) -> List[str]:
    goal = str(learner_profile.get("current_goal", ""))
    words = re.findall(r"[a-zA-ZäöüÄÖÜß]{4,}", goal.lower())
    stopwords = {
        "want", "would", "like", "please", "practice", "learn", "german",
        "deutsch", "today", "help", "with", "about", "improve",
    }
    return [word for word in words if word not in stopwords][:8]


def _state_focus_grammar_points(
    grammar_point: str,
    learner_profile: Optional[Dict[str, Any]],
    grammar_point_mastery: Optional[Dict[str, int]],
) -> List[str]:
    focus = [grammar_point]
    learner_profile = learner_profile or {}
    grammar_point_mastery = grammar_point_mastery or {}

    focus.extend(_as_list(learner_profile.get("recent_grammar_points"))[-3:])
    focus.extend(
        grammar_point
        for grammar_point, score in grammar_point_mastery.items()
        if score <= 1
    )

    seen = set()
    ordered = []
    for item in focus:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered[:4]


def learner_state_score(
    doc,
    target_grammar_point: str,
    learner_profile: Optional[Dict[str, Any]],
    grammar_point_mastery: Optional[Dict[str, int]],
) -> float:
    learner_profile = learner_profile or {}
    grammar_point_mastery = grammar_point_mastery or {}
    metadata = doc.metadata
    doc_topic = str(metadata.get("topic", ""))
    doc_level = str(metadata.get("level", ""))
    doc_grammar_point = str(metadata.get("grammar_point", ""))

    score = 0.0
    if doc_grammar_point == target_grammar_point:
        score += 2.0

    for weak_signal in _as_list(learner_profile.get("weak_topics")):
        if _signal_matches_document(weak_signal, doc):
            score += 1.4

    for recent_topic in _as_list(learner_profile.get("recent_topics")):
        if recent_topic == doc_topic or _signal_matches_document(recent_topic, doc):
            score += 0.8

    for recent_gp in _as_list(learner_profile.get("recent_grammar_points")):
        if recent_gp == doc_grammar_point:
            score += 1.0

    for keyword in _current_goal_keywords(learner_profile):
        if keyword in normalize_text(doc.page_content):
            score += 0.4

    mastery = grammar_point_mastery.get(doc_grammar_point)
    if mastery is not None and mastery <= 1:
        score += 1.2
    elif mastery is not None and mastery >= 3 and doc_grammar_point != target_grammar_point:
        score -= 0.8

    for strong_signal in _as_list(learner_profile.get("strong_topics")):
        if _signal_matches_document(strong_signal, doc) and doc_grammar_point != target_grammar_point:
            score -= 0.6

    if doc_level and any(doc_level in signal for signal in _as_list(learner_profile.get("weak_topics"))):
        score += 0.3

    return score


def rerank_by_learner_state(
    documents,
    grammar_point: str,
    learner_profile: Optional[Dict[str, Any]] = None,
    grammar_point_mastery: Optional[Dict[str, int]] = None,
):
    ranked = []
    for original_rank, doc in enumerate(deduplicate_documents(documents)):
        state_score = learner_state_score(doc, grammar_point, learner_profile, grammar_point_mastery)
        doc.metadata["learner_state_score"] = round(state_score, 3)
        ranked.append((state_score, -original_rank, doc))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _, _, doc in ranked]


def retrieval_focus_summary(
    grammar_point: str,
    learner_profile: Optional[Dict[str, Any]],
    grammar_point_mastery: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    learner_profile = learner_profile or {}
    grammar_point_mastery = grammar_point_mastery or {}
    return {
        "focus_grammar_points": _state_focus_grammar_points(
            grammar_point,
            learner_profile,
            grammar_point_mastery,
        ),
        "weak_topics": _as_list(learner_profile.get("weak_topics"))[-6:],
        "recent_topics": _as_list(learner_profile.get("recent_topics"))[-5:],
        "recent_grammar_points": _as_list(learner_profile.get("recent_grammar_points"))[-5:],
        "low_mastery_grammar_points": [
            grammar_point
            for grammar_point, score in grammar_point_mastery.items()
            if score <= 1
        ],
    }


def retrieve_rule_chunks(
    query: str,
    user_level: str,
    grammar_point: str,
    k: int,
    learner_profile: Optional[Dict[str, Any]] = None,
    grammar_point_mastery: Optional[Dict[str, int]] = None,
):
    documents = []
    focus_grammar_points = _state_focus_grammar_points(grammar_point, learner_profile, grammar_point_mastery)
    candidate_k = max(k * 3, 6)

    for level in LEVEL_FALLBACKS.get(user_level, [user_level]):
        for source in ("LexiPath_ManualRules", "DiscoResearch", "Avemio_ReasoningDE"):
            for focus_grammar_point in focus_grammar_points:
                documents.extend(
                    search_documents(
                        query,
                        {
                            "level": level,
                            "source": source,
                            "chunk_type": "rule",
                            "grammar_point": focus_grammar_point,
                        },
                        candidate_k,
                    )
                )
        if documents:
            break

    return rerank_by_learner_state(documents, grammar_point, learner_profile, grammar_point_mastery)[:k]


def retrieve_example_chunks(
    query: str,
    user_level: str,
    topic: str,
    grammar_point: str,
    k: int,
    learner_profile: Optional[Dict[str, Any]] = None,
    grammar_point_mastery: Optional[Dict[str, int]] = None,
):
    documents = []
    focus_grammar_points = _state_focus_grammar_points(grammar_point, learner_profile, grammar_point_mastery)
    candidate_k = max(k * 3, 8)

    for level in LEVEL_FALLBACKS.get(user_level, [user_level]):
        for focus_grammar_point in focus_grammar_points:
            documents.extend(
                search_documents(
                    query,
                    {
                        "level": level,
                        "source": "Nicos-Weg-GitHub",
                        "chunk_type": "example",
                        "grammar_point": focus_grammar_point,
                    },
                    candidate_k,
                )
            )
        if documents:
            break

    if not documents:
        for level in LEVEL_FALLBACKS.get(user_level, [user_level]):
            documents.extend(
                search_documents(
                    query,
                    {
                        "level": level,
                        "source": "Nicos-Weg-GitHub",
                        "chunk_type": "example",
                        "topic": topic,
                    },
                    k * 2,
                )
            )
            if documents:
                break

    return rerank_by_learner_state(documents, grammar_point, learner_profile, grammar_point_mastery)[:k]


def format_bundle(rule_documents, example_documents, topic: str, grammar_point: str, used_fallback: bool):
    ordered = list(rule_documents) + list(example_documents)
    context_text = "\n\n".join(
        f"[{index}] {doc.page_content}"
        for index, doc in enumerate(ordered, start=1)
    )
    document_summaries: List[Dict[str, str]] = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "topic": doc.metadata.get("topic", topic),
            "level": doc.metadata.get("level", "A1"),
            "chunk_type": doc.metadata.get("chunk_type", "unknown"),
            "grammar_point": doc.metadata.get("grammar_point", grammar_point),
            "learner_state_score": doc.metadata.get("learner_state_score", 0),
            "preview": doc.page_content[:160].replace("\n", " "),
        }
        for doc in ordered
    ]
    return {
        "topic": topic,
        "grammar_point": grammar_point,
        "used_fallback": used_fallback,
        "context_text": context_text,
        "documents": document_summaries,
    }


def retrieve_context_bundle(
    query: str,
    user_level: str,
    topic_hint: str = None,
    k: int = 5,
    learner_profile: Optional[Dict[str, Any]] = None,
    grammar_point_mastery: Optional[Dict[str, int]] = None,
):
    topic = topic_hint or infer_topic(query)
    grammar_point = infer_grammar_point(query, topic)

    rule_documents = retrieve_rule_chunks(
        query,
        user_level,
        grammar_point,
        k=1,
        learner_profile=learner_profile,
        grammar_point_mastery=grammar_point_mastery,
    )
    example_documents = retrieve_example_chunks(
        query,
        user_level,
        topic,
        grammar_point,
        k=max(k - 1, 1),
        learner_profile=learner_profile,
        grammar_point_mastery=grammar_point_mastery,
    )

    used_fallback = False
    if not rule_documents:
        used_fallback = True

    bundle = format_bundle(rule_documents, example_documents, topic, grammar_point, used_fallback)
    bundle["retrieval_focus"] = retrieval_focus_summary(
        grammar_point,
        learner_profile,
        grammar_point_mastery,
    )
    bundle["learner_state_used"] = bool(learner_profile or grammar_point_mastery)
    return bundle


if __name__ == "__main__":
    print(retrieve_context_bundle("How do I use den?", "A1"))
