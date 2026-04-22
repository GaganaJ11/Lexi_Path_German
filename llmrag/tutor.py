from typing import Dict, List

from engine import call_chat
from rag import retrieve, format_context

def tutor_system_prompt(level: str) -> str:
    return f"""
You are a German tutor for a learner at level {level}.

Rules:
- Use the retrieved context when relevant.
- Teach clearly and practically.
- Correct mistakes gently.
- Ask one follow-up at a time.
- Use mostly German, but use English if the learner is confused.

If level is Pre-A1:
- teach absolute beginner German
- use tiny examples
- start with greetings, introducing oneself, very basic verbs

If level is A1:
- teach short everyday German

If level is A2:
- teach connected simple German, Perfekt, weil/deshalb, practical writing

If level is B1:
- teach explanation, opinion, comparison, structured speaking and writing

For every learner reply:
1. acknowledge meaning
2. correct mistakes
3. explain briefly
4. ask one follow-up
"""

def build_retrieval_query(level: str, user_message: str) -> str:
    return f"Learner level: {level}. Learner request: {user_message}"

def start_tutor(level: str):
    print("=" * 60)
    print(f"RAG Tutor started at level: {level}")
    print("Type 'exit' to stop.")
    print("=" * 60)

    first_user_message = "Start teaching me now with a suitable first lesson."
    query = build_retrieval_query(level, first_user_message)
    chunks = retrieve(query, level_filter=None)
    context = format_context(chunks)

    messages: List[Dict] = [
        {"role": "system", "content": tutor_system_prompt(level)},
        {
            "role": "user",
            "content": (
                f"Retrieved context:\n{context}\n\n"
                f"Learner level: {level}\n"
                f"Learner request: {first_user_message}"
            )
        }
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

        query = build_retrieval_query(level, user_input)
        chunks = retrieve(query, level_filter=None)
        context = format_context(chunks)

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