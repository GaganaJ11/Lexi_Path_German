import requests
from typing import Dict, List

from config import OLLAMA_URL, MODEL_NAME, REQUEST_TIMEOUT


def call_ollama(messages: List[Dict]) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]