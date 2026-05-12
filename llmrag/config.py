import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
INDEX_PATH = INDEX_DIR / "lexipath_index.json"

KIMI_API_URL = os.getenv(
    "KIMI_API_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions",
)
KIMI_MODEL_NAME = os.getenv("KIMI_MODEL_NAME", "moonshotai/kimi-k2.6")
KIMI_API_KEY = os.getenv(
    "KIMI_API_KEY",
    "your_moonshot_api_key_here",
)
REQUEST_TIMEOUT = int(os.getenv("KIMI_REQUEST_TIMEOUT", "180"))
CHAT_TEMPERATURE = float(os.getenv("KIMI_TEMPERATURE", "0.3"))
TOP_K = int(os.getenv("LEXIPATH_TOP_K", "4"))

KB_JSONL_CANDIDATES = [
    BASE_DIR / "data" / "LexiPath_Clean_Knowledge_Base.jsonl",
    BASE_DIR / "LexiPath_Clean_Knowledge_Base.jsonl",
]


def resolve_kb_jsonl_path() -> Path:
    override = os.getenv("KB_JSONL_PATH")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        return path

    for candidate in KB_JSONL_CANDIDATES:
        if candidate.exists():
            return candidate

    return KB_JSONL_CANDIDATES[0]


KB_JSONL_PATH = resolve_kb_jsonl_path()
