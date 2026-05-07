import json
from typing import Dict, List

from engine import call_chat, safe_json_load
from prompts import build_diagnostic_intro_prompt
from utils import average_score, is_non_answer

LEVEL_ORDER = ["Pre-A1", "A1", "A2", "B1"]
MAX_SCORE_PER_ATTEMPT = 2
MIN_ATTEMPTS = 3
MAX_ATTEMPTS = 6

LEVEL_BLUEPRINTS = {
    "Pre-A1": [
        {
            "topic": "Greetings",
            "grammar_point": "basic_greeting_phrase",
            "prompt_goal": "Check whether the learner can produce a very short greeting or self-introduction in German.",
            "criteria": "The answer should contain a tiny but meaningful German phrase such as 'Hallo' or 'Ich heiße ...'.",
            "example_answer": "Hallo, ich heiße Anna.",
        },
        {
            "topic": "Basic Answers",
            "grammar_point": "yes_no_basic_response",
            "prompt_goal": "Check whether the learner can answer a very simple everyday question with one short German phrase.",
            "criteria": "The answer should show a basic understandable response such as 'Ja', 'Nein', or one tiny phrase.",
            "example_answer": "Ja, ein bisschen.",
        },
        {
            "topic": "Basic Nouns",
            "grammar_point": "single_word_everyday_vocab",
            "prompt_goal": "Check whether the learner can name one very common everyday object or food item in German.",
            "criteria": "The answer should contain at least one recognizable everyday German noun.",
            "example_answer": "Wasser.",
        },
    ],
    "A1": [
        {
            "topic": "Articles",
            "grammar_point": "indefinite_articles_ein_eine_einen",
            "prompt_goal": "Check whether the learner can produce a simple accusative noun phrase with an indefinite article.",
            "criteria": "The answer should clearly show an accusative masculine phrase such as 'einen Apfel'.",
            "example_answer": "Der Mann isst einen Apfel.",
        },
        {
            "topic": "Negation",
            "grammar_point": "negation_kein",
            "prompt_goal": "Check whether the learner can negate a noun phrase with 'kein'.",
            "criteria": "The answer should use 'kein' or an inflected form like 'keine' correctly.",
            "example_answer": "Nein, ich habe kein Auto.",
        },
        {
            "topic": "Verb Conjugation",
            "grammar_point": "present_tense_basic_verbs",
            "prompt_goal": "Check whether the learner can write one simple present-tense sentence about themselves.",
            "criteria": "The answer should contain a clear present-tense sentence such as 'Ich wohne in Berlin.'",
            "example_answer": "Ich wohne in Berlin.",
        },
    ],
    "A2": [
        {
            "topic": "Verb Conjugation",
            "grammar_point": "perfect_tense_basics",
            "prompt_goal": "Check whether the learner can describe a completed past action with Perfekt.",
            "criteria": "The answer should use a helper verb and a past participle appropriately.",
            "example_answer": "Ich habe gestern Deutsch gelernt.",
        },
        {
            "topic": "Cases",
            "grammar_point": "accusative_with_movement",
            "prompt_goal": "Check whether the learner can use a two-way preposition with movement and accusative.",
            "criteria": "The answer should show movement toward a destination, such as 'auf den Tisch'.",
            "example_answer": "Ich lege das Buch auf den Tisch.",
        },
        {
            "topic": "Grammar",
            "grammar_point": "comparatives_basics",
            "prompt_goal": "Check whether the learner can compare two things with a comparative and 'als'.",
            "criteria": "The answer should include a comparative form plus 'als'.",
            "example_answer": "Ein Auto ist schneller als ein Fahrrad.",
        },
    ],
    "B1": [
        {
            "topic": "Sentence Structure",
            "grammar_point": "subordinate_clause_weil",
            "prompt_goal": "Check whether the learner can produce a 'weil' clause with the verb at the end.",
            "criteria": "The answer should contain a subordinate clause introduced by 'weil' with final verb placement.",
            "example_answer": "Ich lerne Deutsch, weil ich in Deutschland arbeiten will.",
        },
        {
            "topic": "Grammar",
            "grammar_point": "konjunktiv_ii_basics",
            "prompt_goal": "Check whether the learner can express a hypothetical idea with Konjunktiv II.",
            "criteria": "The answer should use a form like 'würde' or another clear Konjunktiv II structure.",
            "example_answer": "Ich würde viel reisen und ein Haus kaufen.",
        },
        {
            "topic": "Sentence Structure",
            "grammar_point": "relative_clauses_basics",
            "prompt_goal": "Check whether the learner can combine two clauses using a relative clause.",
            "criteria": "The answer should use a relative pronoun and a grammatically coherent relative clause.",
            "example_answer": "Das ist die Frau, die mir hilft.",
        },
    ],
}


def _level_index(level: str) -> int:
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        return LEVEL_ORDER.index("A1")


def _clamp_level(level: str) -> str:
    return level if level in LEVEL_ORDER else "A1"


def _extract_json(raw: str, fallback: Dict) -> Dict:
    raw = (raw or "").strip()
    if not raw:
        return fallback

    parsed = safe_json_load(raw, None)
    if isinstance(parsed, dict):
        return parsed

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = safe_json_load(raw[start:end + 1], None)
        if isinstance(parsed, dict):
            return parsed

    return fallback


def _attempt_summary(attempts: List[Dict]) -> str:
    if not attempts:
        return "No attempts yet."

    lines = []
    for attempt in attempts:
        lines.append(
            f"- Attempt {attempt['attempt']}: "
            f"question_level={attempt['question_level']}, "
            f"focus={attempt['focus_topic']}, "
            f"score={attempt['score']}/2, "
            f"estimated_level={attempt['estimated_level']}, "
            f"answer={attempt['answer']!r}, "
            f"feedback={attempt['rationale']}"
        )
    return "\n".join(lines)


def _fallback_question_plan(attempts: List[Dict]) -> Dict[str, str]:
    if not attempts:
        target_level = "A1"
    else:
        last = attempts[-1]
        target_level = _clamp_level(last["estimated_level"])
        if last["score"] == 2 and _level_index(target_level) < len(LEVEL_ORDER) - 1:
            target_level = LEVEL_ORDER[_level_index(target_level) + 1]
        elif last["score"] == 0 and _level_index(target_level) > 0:
            target_level = LEVEL_ORDER[_level_index(target_level) - 1]

    blueprints = LEVEL_BLUEPRINTS[target_level]
    used_points = {
        attempt.get("grammar_point")
        for attempt in attempts
        if attempt["question_level"] == target_level
    }
    blueprint = next(
        (item for item in blueprints if item["grammar_point"] not in used_points),
        blueprints[0],
    )

    return {
        "target_level": target_level,
        "focus_topic": blueprint["topic"],
        "grammar_point": blueprint["grammar_point"],
        "goal": blueprint["prompt_goal"],
        "criteria": blueprint["criteria"],
        "example_answer": blueprint["example_answer"],
    }


def _diagnostic_intro() -> str:
    fallback = (
        "Welcome. Before we begin learning together, I will ask you a few short questions "
        "to understand your current German level. Please answer in German as much as you can. "
        "There is no pressure, and it is completely fine if you are unsure. When you are ready, we can begin."
    )

    try:
        content = call_chat([{"role": "user", "content": build_diagnostic_intro_prompt()}]).strip()
        return content or fallback
    except Exception:
        return fallback


def _plan_next_question(attempts: List[Dict]) -> Dict[str, str]:
    fallback = _fallback_question_plan(attempts)

    prompt = f"""
You are Kimi with RAG, a warm and adaptive German tutor planning the next question for a placement diagnostic.

Available levels: {", ".join(LEVEL_ORDER)}
Level blueprints:
{json.dumps(LEVEL_BLUEPRINTS, ensure_ascii=False, indent=2)}

Previous attempts:
{_attempt_summary(attempts)}

Choose the next best probe level and focus area. Prefer:
- gentle upward movement after strong answers
- same-level confirmation after partial answers
- easier checks after clear failure
- varied topics instead of repeating the same exact grammar point

Return JSON only:
{{
  "target_level": "A1",
  "focus_topic": "Verb Conjugation",
  "grammar_point": "present_tense_basic_verbs",
  "goal": "one short sentence explaining what this question should check",
  "criteria": "one short sentence explaining what a good answer should contain",
  "example_answer": "one short example answer"
}}
""".strip()

    try:
        planned = _extract_json(call_chat([{"role": "user", "content": prompt}]), fallback)
        return {
            "target_level": _clamp_level(planned.get("target_level", fallback["target_level"])),
            "focus_topic": planned.get("focus_topic", fallback["focus_topic"]).strip() or fallback["focus_topic"],
            "grammar_point": planned.get("grammar_point", fallback["grammar_point"]).strip() or fallback["grammar_point"],
            "goal": planned.get("goal", fallback["goal"]).strip() or fallback["goal"],
            "criteria": planned.get("criteria", fallback["criteria"]).strip() or fallback["criteria"],
            "example_answer": planned.get("example_answer", fallback["example_answer"]).strip() or fallback["example_answer"],
        }
    except Exception:
        return fallback


def _generate_question(plan: Dict[str, str], attempts: List[Dict]) -> str:
    prompt = f"""
You are Kimi with RAG, a warm, human-like German tutor writing one short placement question.

Target level: {plan['target_level']}
Focus topic: {plan['focus_topic']}
Grammar point: {plan['grammar_point']}
Goal: {plan['goal']}
Success criteria: {plan['criteria']}
Reference answer: {plan['example_answer']}

Previous attempts summary:
{_attempt_summary(attempts)}

Rules:
- Ask for an answer in German.
- Use English when giving the instruction to the learner.
- Use short, clear teacher wording.
- Keep it practical and natural.
- Make the prompt fit the target level.
- Sound supportive, calm, and not robotic.
- Do not include the answer.
- Do not mention CEFR labels in the question itself.

Reply with only the question.
""".strip()

    try:
        content = call_chat([{"role": "user", "content": prompt}]).strip()
        return content or f"Please answer in German about {plan['focus_topic']}."
    except Exception:
        return f"Please answer in German about {plan['focus_topic']}."


def _fallback_grade(plan: Dict[str, str], answer: str) -> Dict[str, str]:
    text = f" {answer.lower()} "
    word_count = len(answer.split())

    if is_non_answer(answer):
        return {
            "label": "FAIL",
            "score": 0,
            "estimated_level": "Pre-A1",
            "rationale": "There was not enough German to evaluate.",
        }

    signal_map = {
        "Pre-A1": [" ja ", " nein ", " ich ", " du "],
        "A1": [" ich ", " ist ", " habe ", " bin ", " kein ", " eine ", " einen "],
        "A2": [" habe ", " bin ", " weil ", " als ", " auf den ", " in den "],
        "B1": [" obwohl ", " würde ", " dass ", " die ", " der ", " meiner meinung "],
    }

    level = plan["target_level"]
    markers = signal_map.get(level, [])
    hits = sum(1 for marker in markers if marker in text)

    if word_count <= 2:
        return {
            "label": "FAIL",
            "score": 0,
            "estimated_level": "Pre-A1",
            "rationale": "The response is too short for the target task.",
        }
    if hits >= 2 or (level in {"A1", "A2"} and word_count >= 5):
        return {
            "label": "FULL",
            "score": 2,
            "estimated_level": level,
            "rationale": "The answer shows the target pattern clearly enough.",
        }
    if hits >= 1 or word_count >= 4:
        estimated_level = level if level != "Pre-A1" else "A1"
        return {
            "label": "PARTIAL",
            "score": 1,
            "estimated_level": estimated_level,
            "rationale": "The answer is understandable but only partly meets the target.",
        }

    lower_index = max(0, _level_index(level) - 1)
    return {
        "label": "FAIL",
        "score": 0,
        "estimated_level": LEVEL_ORDER[lower_index],
        "rationale": "The target grammar or sentence control is not clear yet.",
    }


def _grade_answer(plan: Dict[str, str], question: str, answer: str, attempts: List[Dict]) -> Dict[str, str]:
    fallback = _fallback_grade(plan, answer)

    prompt = f"""
You are Kimi with RAG, a warm and careful German tutor grading one answer from an adaptive placement diagnostic.

Target level: {plan['target_level']}
Focus topic: {plan['focus_topic']}
Grammar point: {plan['grammar_point']}
Goal: {plan['goal']}
Criteria: {plan['criteria']}
Reference answer: {plan['example_answer']}
Question: {question}
Learner answer: {answer}

Previous attempts summary:
{_attempt_summary(attempts)}

Grade the answer based on grammatical control, relevance to the prompt, and how much support the learner would need.

Return JSON only:
{{
  "label": "FULL",
  "score": 2,
  "estimated_level": "A2",
  "rationale": "one short sentence"
}}

Rules:
- label must be FULL, PARTIAL, or FAIL
- score must be 2, 1, or 0
- estimated_level must be one of {LEVEL_ORDER}
- rationale must be short, kind, and human-sounding
""".strip()

    try:
        graded = _extract_json(call_chat([{"role": "user", "content": prompt}]), fallback)
        label = str(graded.get("label", fallback["label"])).upper()
        score = graded.get("score", fallback["score"])
        estimated_level = _clamp_level(str(graded.get("estimated_level", fallback["estimated_level"])))
        rationale = str(graded.get("rationale", fallback["rationale"])).strip() or fallback["rationale"]

        if label not in {"FULL", "PARTIAL", "FAIL"} or score not in {0, 1, 2}:
            return fallback

        return {
            "label": label,
            "score": score,
            "estimated_level": estimated_level,
            "rationale": rationale,
        }
    except Exception:
        return fallback


def _fallback_decision(attempts: List[Dict]) -> Dict[str, str]:
    if not attempts:
        return {"action": "continue", "final_level": "A1", "reason": "Need more evidence."}

    scores = [attempt["score"] for attempt in attempts]
    avg_score = average_score({str(i): score for i, score in enumerate(scores, start=1)})
    strong_b1 = any(
        attempt["question_level"] == "B1" and attempt["score"] == 2
        for attempt in attempts
    )
    strong_a2 = any(
        attempt["question_level"] == "A2" and attempt["score"] >= 1
        for attempt in attempts
    )
    weak_a1 = sum(
        1 for attempt in attempts
        if attempt["question_level"] in {"Pre-A1", "A1"} and attempt["score"] == 0
    ) >= 2

    if len(attempts) < MIN_ATTEMPTS:
        return {"action": "continue", "final_level": "A1", "reason": "Need more evidence."}

    if strong_b1 and avg_score >= 1.2:
        final_level = "B1"
    elif strong_a2 and avg_score >= 1.0:
        final_level = "A2"
    elif weak_a1 and avg_score < 0.8:
        final_level = "Pre-A1"
    else:
        final_level = "A1"

    if len(attempts) >= MAX_ATTEMPTS or avg_score < 0.6 or strong_b1:
        return {"action": "stop", "final_level": final_level, "reason": "Enough evidence collected."}

    recent_partial = len(attempts) >= 2 and all(attempt["score"] == 1 for attempt in attempts[-2:])
    if recent_partial:
        return {"action": "stop", "final_level": final_level, "reason": "Performance has stabilized."}

    return {"action": "continue", "final_level": final_level, "reason": "Collect one more sample."}


def _decide_next_step(attempts: List[Dict]) -> Dict[str, str]:
    fallback = _fallback_decision(attempts)

    prompt = f"""
You are deciding whether to continue or stop an adaptive German placement diagnostic.

Attempts so far:
{_attempt_summary(attempts)}

Rules:
- Minimum attempts before stopping: {MIN_ATTEMPTS}
- Maximum attempts: {MAX_ATTEMPTS}
- Stop when the learner's level is clear enough.
- Final level must be one of {LEVEL_ORDER}.
- Avoid overtesting.

Return JSON only:
{{
  "action": "continue",
  "final_level": "A1",
  "reason": "one short sentence"
}}
""".strip()

    try:
        decision = _extract_json(call_chat([{"role": "user", "content": prompt}]), fallback)
        action = str(decision.get("action", fallback["action"])).lower()
        final_level = _clamp_level(str(decision.get("final_level", fallback["final_level"])))
        reason = str(decision.get("reason", fallback["reason"])).strip() or fallback["reason"]

        if action not in {"continue", "stop"}:
            return fallback

        if len(attempts) < MIN_ATTEMPTS:
            action = "continue"
        if len(attempts) >= MAX_ATTEMPTS:
            action = "stop"

        return {
            "action": action,
            "final_level": final_level,
            "reason": reason,
        }
    except Exception:
        return fallback


def _format_question(plan: Dict[str, str], generated_question: str) -> str:
    return generated_question


def _build_completion_message(final_level: str, attempts: List[Dict]) -> str:
    total_points = sum(attempt["score"] for attempt in attempts)
    max_points = len(attempts) * MAX_SCORE_PER_ATTEMPT
    evidence = ", ".join(
        f"{attempt['question_level']}:{attempt['score']}/2"
        for attempt in attempts
    )

    return (
        f"Thanks for working through that with me. "
        f"I'd place you around {final_level} right now. "
        f"You scored {total_points}/{max_points} across {len(attempts)} adaptive checks "
        f"({evidence}). "
        f"From here, I'll adjust my explanations so they feel manageable and useful for you."
    )


def run_diagnosis() -> str:
    print()
    print(f"Tutor: {_diagnostic_intro()}")
    print()

    attempts: List[Dict] = []

    while len(attempts) < MAX_ATTEMPTS:
        plan = _plan_next_question(attempts)
        question = _generate_question(plan, attempts)
        prompt = _format_question(plan, question)

        print(f"\nTutor: {prompt}")
        answer = input("You: ").strip()

        if answer.lower() in {"exit", "quit", "stop"}:
            print("Session ended.")
            return "A1"

        grade = _grade_answer(plan, question, answer, attempts)
        attempt = {
            "attempt": len(attempts) + 1,
            "question_level": plan["target_level"],
            "focus_topic": plan["focus_topic"],
            "grammar_point": plan["grammar_point"],
            "goal": plan["goal"],
            "question": question,
            "answer": answer,
            "label": grade["label"],
            "score": grade["score"],
            "estimated_level": grade["estimated_level"],
            "rationale": grade["rationale"],
        }
        attempts.append(attempt)

        print(f"Tutor: {grade['rationale']}")

        decision = _decide_next_step(attempts)
        if decision["action"] == "stop":
            final_level = decision["final_level"]
            completion = _build_completion_message(final_level, attempts)
            print(f"\nTutor: {completion}")
            print(f"Detected level: {final_level}")
            return final_level

    final_level = _fallback_decision(attempts)["final_level"]
    completion = _build_completion_message(final_level, attempts)

    print(f"\nTutor: {completion}")
    print(f"Detected level: {final_level}")
    return final_level
