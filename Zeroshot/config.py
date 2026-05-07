import os


KIMI_API_URL = os.getenv(
    "KIMI_API_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions",
)
KIMI_MODEL_NAME = os.getenv("KIMI_MODEL_NAME", "moonshotai/kimi-k2.6")
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "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")
REQUEST_TIMEOUT = int(os.getenv("KIMI_REQUEST_TIMEOUT", "180"))

