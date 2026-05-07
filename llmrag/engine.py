import json
from typing import Dict, List, Optional

import requests

from config import CHAT_TEMPERATURE, KIMI_API_KEY, KIMI_API_URL, KIMI_MODEL_NAME, REQUEST_TIMEOUT


def _headers() -> Dict[str, str]:
    if not KIMI_API_KEY:
        raise RuntimeError(
            "KIMI_API_KEY is not configured. Set it in your environment before running the app."
        )

    return {
        "Authorization": f"Bearer {KIMI_API_KEY}",
        "Content-Type": "application/json",
    }


def _extract_message_content(data: Dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Kimi response did not include choices: {data}")

    message = choices[0].get("message") or {}
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def call_chat(messages: List[Dict], json_schema: Optional[Dict] = None) -> str:
    payload: Dict = {
        "model": KIMI_MODEL_NAME,
        "messages": messages,
        "temperature": CHAT_TEMPERATURE,
        "stream": False,
    }

    if json_schema is not None:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        KIMI_API_URL,
        headers=_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    content = _extract_message_content(data)
    if not content:
        raise RuntimeError(f"Kimi returned an empty message: {data}")
    return content


def safe_json_load(raw: str, fallback: Optional[Dict]) -> Optional[Dict]:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return fallback
