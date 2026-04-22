import json
import requests
from typing import Dict, List, Optional

from config import (
    OLLAMA_CHAT_URL,
    OLLAMA_EMBED_URL,
    CHAT_MODEL,
    EMBED_MODEL,
    REQUEST_TIMEOUT
)

def call_chat(messages: List[Dict], json_schema: Optional[Dict] = None) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False
    }

    if json_schema is not None:
        payload["format"] = json_schema

    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]

def embed_texts(texts: List[str]) -> List[List[float]]:
    payload = {
        "model": EMBED_MODEL,
        "input": texts
    }

    response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    return data["embeddings"]

def safe_json_load(raw: str, fallback: Dict) -> Dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback