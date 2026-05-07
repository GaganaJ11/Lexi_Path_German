import json
from collections import Counter, defaultdict
from typing import Dict, List

from config import INDEX_DIR, INDEX_PATH, KB_JSONL_PATH
from utils import tokenize


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_row(row: Dict) -> Dict:
    text_value = row.get("text") or row.get("content") or ""
    metadata = row.get("metadata", {})
    return {
        "text": text_value.strip(),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def build_index() -> None:
    print(f"Loading cleaned JSONL from {KB_JSONL_PATH}...")
    rows = load_jsonl(str(KB_JSONL_PATH))
    print(f"Loaded {len(rows)} records")

    docs = []
    inverted_index = defaultdict(list)
    doc_frequencies = Counter()
    total_length = 0

    for row in rows:
        normalized = _normalize_row(row)
        if not normalized["text"]:
            continue

        tokens = tokenize(normalized["text"])
        if not tokens:
            continue

        term_counts = Counter(tokens)
        doc_id = len(docs)
        total_length += len(tokens)

        docs.append(
            {
                "id": doc_id,
                "text": normalized["text"],
                "metadata": normalized["metadata"],
                "length": len(tokens),
            }
        )

        for term in term_counts:
            doc_frequencies[term] += 1

        for term, frequency in term_counts.items():
            inverted_index[term].append([doc_id, frequency])

    if not docs:
        raise ValueError(f"No valid text chunks found in {KB_JSONL_PATH}.")

    index_payload = {
        "doc_count": len(docs),
        "avg_doc_len": total_length / len(docs),
        "docs": docs,
        "doc_frequencies": dict(doc_frequencies),
        "inverted_index": dict(inverted_index),
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(index_payload, file_obj, ensure_ascii=False)

    print(f"Saved lexical index to {INDEX_PATH}")
    print(f"Total indexed chunks: {len(docs)}")


if __name__ == "__main__":
    build_index()
