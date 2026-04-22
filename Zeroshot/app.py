from diagnostic_logic import run_diagnostic
from tutor import start_tutor


def print_user_friendly_summary(summary: dict) -> None:
    level = summary["detected_level"]
    strengths = summary.get("strengths", [])
    weaknesses = summary.get("weaknesses", [])

    

    print(f"Your current German level is: {level}")
    print()

    if level == "Pre-A1":
        print("This means you are just starting German, and we will begin from the basics.")
    elif level == "A1":
        print("This means you already know some basics, and we will build a stronger foundation.")
    elif level == "A2":
        print("This means you can handle everyday German, and we will now improve your confidence and fluency.")
    else:
        print("This means you are already at an intermediate level, and we can work on clearer and more natural German.")

    print()

    if strengths:
        print("What you are already doing well:")
        for s in strengths:
            print(f"- {s}")
        print()

    if weaknesses:
        print("What we will improve next:")
        for w in weaknesses:
            print(f"- {w}")
        print()

    print("We will now continue with a structured learning plan.")
    print()


def main():
    results, summary = run_diagnostic()

    print_user_friendly_summary(summary)

    final_level = summary["detected_level"]
    start_tutor(final_level, summary)


if __name__ == "__main__":
    main()