"""
Single-place configuration for LexiPath German.

For local use, edit the two values in the "USER SETTINGS" section below.
Do not share or commit this file after adding a real API key.
"""

# =========================
# USER SETTINGS
# =========================

NVIDIA_API_KEY = "sk-JOcL5kI8nLSisFiefomELDGK3uLBONHssHbntgGjNh5schmA"
DATABASE_URL = "postgresql+psycopg://postgres:mypassword@localhost:5432/postgres"


# =========================
# DEFAULT PROJECT SETTINGS
# Usually you do not need to change these.
# =========================

MOONSHOT_API_KEY = NVIDIA_API_KEY
MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
MOONSHOT_MODEL = "kimi-k2.5"
MOONSHOT_TEMPERATURE = 1
MOONSHOT_TOP_P = 0.95
MOONSHOT_MAX_TOKENS = 1200
MOONSHOT_TIMEOUT_SECONDS = 90

COLLECTION_NAME = "lexipath_grammar_v2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CORPUS_MODE = "paper_curated"
PAPER_ALIGNED_CORPUS_SIZE = 1240
PAPER_RULE_TARGET = 300


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("your_") or "your_" in value


def get_moonshot_api_key() -> str:
    if _is_placeholder(MOONSHOT_API_KEY):
        raise RuntimeError(
            "Missing Moonshot API key. Open LexiPath_German/config.py and replace "
            "NVIDIA_API_KEY or MOONSHOT_API_KEY with your real Moonshot key."
        )
    return MOONSHOT_API_KEY


def get_database_url() -> str:
    if _is_placeholder(DATABASE_URL):
        raise RuntimeError(
            "Missing database URL. Open LexiPath_German/config.py and replace "
            "DATABASE_URL with your local Postgres/pgvector URL."
        )
    return DATABASE_URL
