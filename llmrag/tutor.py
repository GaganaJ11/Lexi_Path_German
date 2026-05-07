from typing import Dict, List

from engine import call_chat
from prompts import build_tutor_starter_prompt, tutor_system_prompt
from rag import retrieve, format_context

def build_retrieval_query(level: str, user_message: str) -> str:
    return f"Learner level: {level}. Learner request: {user_message}"


def retrieve_context(level: str, user_message: str) -> str:
    query = build_retrieval_query(level, user_message)
    preferred_level = None if level == "Pre-A1" else level
    chunks = retrieve(query, level_filter=preferred_level)
    if not chunks:
        chunks = retrieve(query, level_filter=None)
    return format_context(chunks)


def start_tutor(level: str):

    first_user_message = "Start teaching me now with a suitable first lesson."
    context = retrieve_context(level, first_user_message)
    starter_message = build_tutor_starter_prompt(level, context)

    messages: List[Dict] = [
        {"role": "system", "content": tutor_system_prompt(level)},
        {"role": "user", "content": starter_message},
    ]

    reply = call_chat(messages)
    messages.append({"role": "assistant", "content": reply})

    print("\nTutor:")
    print(reply)
    print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "stop"}:
            print("Session ended.")
            break

        context = retrieve_context(level, user_input)

        messages.append({
            "role": "user",
            "content": (
                f"Retrieved context:\n{context}\n\n"
                f"Learner says: {user_input}"
            )
        })

        reply = call_chat(messages)
        messages.append({"role": "assistant", "content": reply})

        print("\nTutor:")
        print(reply)
        print()
