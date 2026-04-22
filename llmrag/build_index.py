import json
import os
from typing import List, Dict

from config import KB_JSONL_PATH, INDEX_PATH
from engine import embed_texts

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_index():
    print("Loading cleaned JSONL...")
    rows = load_jsonl(KB_JSONL_PATH)
    print(f"Loaded {len(rows)} records")

    texts = []
    normalized_rows = []
    for row in rows:
        text_value = row.get("text") or row.get("content")
        if not text_value:
            continue
        texts.append(text_value)
        normalized_rows.append(
            {
                "text": text_value,
                "metadata": row.get("metadata", {}),
            }
        )

    if not normalized_rows:
        raise ValueError(
            f"No valid text chunks found in {KB_JSONL_PATH}. Expected keys 'text' or 'content'."
        )

    print("Generating embeddings...")
    embeddings = embed_texts(texts)

    indexed = []
    for row, emb in zip(normalized_rows, embeddings):
        indexed.append({
            "text": row["text"],
            "metadata": row.get("metadata", {}),
            "embedding": emb
        })

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(indexed, f, ensure_ascii=False)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Total indexed chunks: {len(indexed)}")

if __name__ == "__main__":
    build_index()
