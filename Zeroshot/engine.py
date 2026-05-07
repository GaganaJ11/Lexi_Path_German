import requests
from typing import Dict, List

from config import KIMI_API_KEY, KIMI_API_URL, KIMI_MODEL_NAME, REQUEST_TIMEOUT


def call_kimi(messages: List[Dict]) -> str:
    if not KIMI_API_KEY:
        raise RuntimeError(
            "Missing KIMI_API_KEY. Set it in config.py or your environment before running the app."
        )

    payload = {
        "model": KIMI_MODEL_NAME,
        "messages": messages,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {KIMI_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        KIMI_API_URL,
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]
