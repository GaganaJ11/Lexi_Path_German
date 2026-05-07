import json
import math
import re
from collections import Counter
from typing import Dict, Iterable, List


GERMAN_STOPWORDS = {
    "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "bin", "bis", "bist",
    "da", "dann", "das", "dass", "dein", "dem", "den", "der", "des", "die", "dir",
    "doch", "dort", "du", "ein", "eine", "einem", "einen", "einer", "er", "es",
    "für", "hat", "hast", "haben", "hier", "ich", "ihr", "ihn", "im", "in", "ist",
    "ja", "kein", "keine", "mit", "mich", "mir", "nicht", "oder", "sehr", "sein",
    "sie", "sind", "so", "und", "uns", "von", "war", "was", "weil", "wenn", "wie",
    "wir", "wo", "zu",
}


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
        "no",
    }
    return cleaned in bad or len(cleaned) <= 1


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-zÀ-ÿÄÖÜäöüß][A-Za-zÀ-ÿÄÖÜäöüß0-9_-]*", text.lower())


def normalize_query_terms(text: str) -> List[str]:
    tokens = tokenize(text)
    filtered = [token for token in tokens if token not in GERMAN_STOPWORDS and len(token) > 1]
    return filtered or tokens[:8]


def term_frequencies(tokens: Iterable[str]) -> Dict[str, int]:
    return dict(Counter(tokens))
