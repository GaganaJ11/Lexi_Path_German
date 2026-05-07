"""
Single-place configuration for LexiPath German.

For local use, edit the two values in the "USER SETTINGS" section below.
Do not share or commit this file after adding a real API key.
"""

# =========================
# USER SETTINGS
# =========================

NVIDIA_API_KEY = "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"
DATABASE_URL = "postgresql+psycopg://postgres:mypassword@localhost:5432/postgres"


# =========================
# DEFAULT PROJECT SETTINGS
# Usually you do not need to change these.
# =========================

NVIDIA_MODEL = "moonshotai/kimi-k2.6"
NVIDIA_TEMPERATURE = 1
NVIDIA_TOP_P = 1
NVIDIA_MAX_COMPLETION_TOKENS = 2048

COLLECTION_NAME = "lexipath_grammar_v2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CORPUS_MODE = "paper_curated"
PAPER_ALIGNED_CORPUS_SIZE = 1240
PAPER_RULE_TARGET = 300


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("your_") or "your_" in value


def get_nvidia_api_key() -> str:
    if _is_placeholder(NVIDIA_API_KEY):
        raise RuntimeError(
            "Missing NVIDIA API key. Open LexiPath_German/config.py and replace "
            "NVIDIA_API_KEY with your real nvapi key."
        )
    return NVIDIA_API_KEY


def get_database_url() -> str:
    if _is_placeholder(DATABASE_URL):
        raise RuntimeError(
            "Missing database URL. Open LexiPath_German/config.py and replace "
            "DATABASE_URL with your local Postgres/pgvector URL."
        )
    return DATABASE_URL
