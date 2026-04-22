import json
import math
from typing import Dict, List

def pretty_json(data: Dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)

def average_score(score_dict: Dict[str, int]) -> float:
    vals = list(score_dict.values())
    return sum(vals) / len(vals) if vals else 0.0

def is_non_answer(answer: str) -> bool:
    if not answer:
        return True

    cleaned = answer.strip().lower()
    bad = {
        "",
        "i don't know",
        "dont know",
        "don't know",
        "idk",
        "skip",
        "only english",
        "english please",
        "no german",
        "i can't speak german",
        "i cannot speak german",
        "none",
        "no"
    }
    return cleaned in bad or len(cleaned) <= 1