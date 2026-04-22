from typing import Dict, List

def average_score(score_dict: Dict[str, int]) -> float:
    values = list(score_dict.values())
    return sum(values) / len(values) if values else 0.0

def is_non_answer(answer: str) -> bool:
    if not answer:
        return True

    cleaned = answer.strip().lower()

    non_answers = {
        "",
        "i don't know",
        "dont know",
        "don't know",
        "idk",
        "skip",
        "no german",
        "only english",
        "english please",
        "i can't speak german",
        "i cannot speak german",
        "i do not know german",
        "none",
        "no"
    }

    if cleaned in non_answers:
        return True

    if len(cleaned) <= 1:
        return True

    return False

def pretty_json(data: Dict) -> str:
    import json
    return json.dumps(data, indent=2, ensure_ascii=False)

def band_to_rank(band: str) -> int:
    mapping = {
        "Pre-A1": 0,
        "A1": 1,
        "A2": 2,
        "B1": 3
    }
    return mapping.get(band, 0)