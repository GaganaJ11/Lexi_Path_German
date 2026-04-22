from typing import Dict, List
from syllabus import LEVEL_ORDER, format_syllabus_for_learner
from engine import call_ollama
from prompts import tutor_system_prompt


def _get_next_level(level: str) -> str:
    if level not in LEVEL_ORDER:
        return LEVEL_ORDER[0]
    idx = LEVEL_ORDER.index(level)
    if idx + 1 >= len(LEVEL_ORDER):
        return level
    return LEVEL_ORDER[idx + 1]


def start_tutor(level: str, summary: Dict) -> None:
    print()
    print("Let's start your learning journey 😊")
    print()

    print(f"You are currently around: {level}")
    print()

    print("Here is your structured learning plan:")
    print()

    formatted_syllabus = format_syllabus_for_learner(level)
    print(formatted_syllabus)
    print()

    if summary.get("strengths"):
        print("What you are already doing well:")
        for s in summary["strengths"]:
            print(f"- {s}")

    if summary.get("weaknesses"):
        print("\nWhat we can improve next:")
        for w in summary["weaknesses"]:
            print(f"- {w}")

    print()
    print("We can learn in a flexible way.")
    print("- follow the full structured plan")
    print("- skip parts you already know")
    print("- revise current level before moving ahead")
    print("- proceed to the next level when ready")
    print("- focus on speaking, grammar, writing, work, or daily conversation")
    print()

    user_choice = input(
        "What would you like to do first? "
        "(Example: full plan / revise current level / proceed next level / focus on speaking): "
    ).strip()
    next_level = _get_next_level(level)

    starter_message = f"""
The learner has completed a diagnosis.

Detected level: {level}
Strengths: {summary.get('strengths', [])}
Weaknesses: {summary.get('weaknesses', [])}

Structured syllabus:
{formatted_syllabus}

Learner preference:
{user_choice}

Please act as a warm, human-like private German tutor.

Important:
- explain the learner's level in a friendly way
- appreciate the learner after every answer
- encourage them even when there are mistakes
- follow the structured syllabus step by step unless the learner wants otherwise
- after current level mastery, move to the next CEFR level automatically (A1 -> A2 -> B1)
- never restart a finished level unless the learner asks to revise it
- clearly mention the current level, section, and topic
- teach one small topic at a time
- after each topic, ask whether they want more examples or want to continue
- when giving choices, include:
  1) revise the current level
  2) proceed to the next level ({next_level})
- if they did well, tell them positively
- do not sound robotic

In your first reply:
1. explain the learner's level in a warm way
2. encourage them
3. respond to their preference
4. introduce the structured syllabus briefly
5. start with the first module and the first topic unless the learner asked to skip ahead
6. ask one small comfortable first exercise
7. mention that once {level} is complete, we continue with {next_level}
"""

    messages: List[Dict] = [
        {"role": "system", "content": tutor_system_prompt(level)},
        {"role": "user", "content": starter_message}
    ]

    reply = call_ollama(messages)
    messages.append({"role": "assistant", "content": reply})

    print("\nTutor:")
    print(reply)
    print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "stop"}:
            print("Session ended.")
            break

        messages.append({"role": "user", "content": user_input})
        reply = call_ollama(messages)
        messages.append({"role": "assistant", "content": reply})

        print("\nTutor:")
        print(reply)
        print()
