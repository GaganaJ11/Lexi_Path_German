import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
LEARNER_STORE_PATH = BASE_DIR / "learners.json"


def _ensure_store_file() -> None:
    if not LEARNER_STORE_PATH.exists():
        LEARNER_STORE_PATH.write_text("{}", encoding="utf-8")


def load_all_learners() -> Dict[str, Dict[str, Any]]:
    _ensure_store_file()
    try:
        with open(LEARNER_STORE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_all_learners(data: Dict[str, Dict[str, Any]]) -> None:
    _ensure_store_file()
    with open(LEARNER_STORE_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def normalize_learner_id(learner_id: str) -> str:
    return learner_id.strip().lower()


def learner_exists(learner_id: str) -> bool:
    learners = load_all_learners()
    return normalize_learner_id(learner_id) in learners


def load_learner(learner_id: str) -> Optional[Dict[str, Any]]:
    learners = load_all_learners()
    return learners.get(normalize_learner_id(learner_id))


def save_learner(learner_id: str, learner_data: Dict[str, Any]) -> None:
    learners = load_all_learners()
    key = normalize_learner_id(learner_id)

    payload = dict(learner_data)
    payload["learner_id"] = key
    payload["display_name"] = learner_data.get("display_name", learner_id.strip())
    payload["last_active"] = datetime.utcnow().isoformat()

    learners[key] = payload
    save_all_learners(learners)


def delete_learner(learner_id: str) -> None:
    learners = load_all_learners()
    key = normalize_learner_id(learner_id)
    if key in learners:
        del learners[key]
        save_all_learners(learners)


def build_learner_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_level": state.get("user_level", "Unknown"),
        "diagnostic_results": state.get("diagnostic_results", {}),
        "diagnostic_feedback": state.get("diagnostic_feedback", []),
        "learner_profile": state.get("learner_profile", {}),
        "grammar_point_mastery": state.get("grammar_point_mastery", {}),
        "phase": state.get("phase", "diagnostic"),
        "intro_shown": state.get("intro_shown", False),
    }
