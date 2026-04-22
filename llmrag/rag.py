import json
from typing import List, Dict, Optional

from config import INDEX_PATH, TOP_K
from engine import embed_texts
from utils import cosine_similarity

def load_index() -> List[Dict]:
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve(query: str, top_k: int = TOP_K, level_filter: Optional[str] = None) -> List[Dict]:
    index = load_index()
    query_embedding = embed_texts([query])[0]

    candidates = []

    for item in index:
        metadata = item.get("metadata", {})

        if level_filter is not None:
            item_level = metadata.get("level")
            if item_level != level_filter:
                continue

        score = cosine_similarity(query_embedding, item["embedding"])

        candidates.append({
            "score": score,
            "text": item["text"],
            "metadata": metadata
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

def format_context(chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant learning context found."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        md = chunk.get("metadata", {})
        source = md.get("source", "unknown")
        level = md.get("level", "unknown")
        skill = md.get("skill", "unknown")
        topic = md.get("topic", "unknown")

        parts.append(
            f"[Chunk {i}]\n"
            f"Source: {source}\n"
            f"Level: {level}\n"
            f"Skill: {skill}\n"
            f"Topic: {topic}\n"
            f"Text: {chunk['text']}"
        )

    return "\n\n".join(parts)