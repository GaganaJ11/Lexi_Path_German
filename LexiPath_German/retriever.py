import re
from functools import lru_cache
from typing import Dict, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

CONNECTION_STRING = "postgresql+psycopg://postgres:mypassword@localhost:5432/postgres"
COLLECTION_NAME = "lexipath_grammar_v2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
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
        connection=CONNECTION_STRING,
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
        key = (doc.page_content, tuple(sorted(doc.metadata.items())))
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def retrieve_rule_chunks(query: str, user_level: str, grammar_point: str, k: int):
    documents = []

    for level in LEVEL_FALLBACKS.get(user_level, [user_level]):
        for source in ("LexiPath_ManualRules", "DiscoResearch", "Avemio_ReasoningDE"):
            documents.extend(
                search_documents(
                    query,
                    {
                        "level": level,
                        "source": source,
                        "chunk_type": "rule",
                        "grammar_point": grammar_point,
                    },
                    k,
                )
            )
        if documents:
            break

    return deduplicate_documents(documents)[:k]


def retrieve_example_chunks(query: str, user_level: str, topic: str, grammar_point: str, k: int):
    documents = []

    for level in LEVEL_FALLBACKS.get(user_level, [user_level]):
        documents.extend(
            search_documents(
                query,
                {
                    "level": level,
                    "source": "Nicos-Weg-GitHub",
                    "chunk_type": "example",
                    "grammar_point": grammar_point,
                },
                k * 2,
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

    return deduplicate_documents(documents)[:k]


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


def retrieve_context_bundle(query: str, user_level: str, topic_hint: str = None, k: int = 4):
    topic = topic_hint or infer_topic(query)
    grammar_point = infer_grammar_point(query, topic)

    rule_documents = retrieve_rule_chunks(query, user_level, grammar_point, k=1)
    example_documents = retrieve_example_chunks(query, user_level, topic, grammar_point, k=max(k - 1, 1))

    used_fallback = False
    if not rule_documents:
        used_fallback = True

    return format_bundle(rule_documents, example_documents, topic, grammar_point, used_fallback)


if __name__ == "__main__":
    print(retrieve_context_bundle("How do I use den?", "A1"))
