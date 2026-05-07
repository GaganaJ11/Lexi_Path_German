import json
import math
from functools import lru_cache
from typing import Dict, List, Optional

from config import INDEX_PATH, TOP_K
from utils import normalize_query_terms


@lru_cache(maxsize=1)
def load_index() -> Dict:
    with open(INDEX_PATH, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _idf(term: str, index: Dict) -> float:
    doc_count = max(index.get("doc_count", 0), 1)
    df = index.get("doc_frequencies", {}).get(term, 0)
    return math.log(1 + (doc_count - df + 0.5) / (df + 0.5))


def retrieve(query: str, top_k: int = TOP_K, level_filter: Optional[str] = None) -> List[Dict]:
    index = load_index()
    query_terms = normalize_query_terms(query)
    if not query_terms:
        return []

    avg_doc_len = index.get("avg_doc_len", 1.0) or 1.0
    k1 = 1.5
    b = 0.75
    scores: Dict[int, float] = {}

    for term in query_terms:
        postings = index.get("inverted_index", {}).get(term, [])
        idf = _idf(term, index)

        for doc_id, term_frequency in postings:
            doc = index["docs"][doc_id]
            metadata = doc.get("metadata", {})
            if level_filter is not None and metadata.get("level") != level_filter:
                continue

            doc_length = max(doc.get("length", 0), 1)
            numerator = term_frequency * (k1 + 1)
            denominator = term_frequency + k1 * (1 - b + b * (doc_length / avg_doc_len))
            score = idf * (numerator / denominator)

            if metadata.get("level") == level_filter:
                score += 0.15

            scores[doc_id] = scores.get(doc_id, 0.0) + score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results = []
    for doc_id, score in ranked:
        doc = index["docs"][doc_id]
        results.append(
            {
                "score": round(score, 4),
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
            }
        )
    return results


def format_context(chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant learning context found."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        level = metadata.get("level", "unknown")
        skill = metadata.get("skill", "unknown")
        topic = metadata.get("topic", "unknown")

        parts.append(
            f"[Chunk {i}]\n"
            f"Score: {chunk.get('score', 0.0)}\n"
            f"Source: {source}\n"
            f"Level: {level}\n"
            f"Skill: {skill}\n"
            f"Topic: {topic}\n"
            f"Text: {chunk['text']}"
        )

    return "\n\n".join(parts)
