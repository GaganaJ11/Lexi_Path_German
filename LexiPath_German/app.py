import re
import os
from typing import Any, Dict, List, TypedDict

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph

from diagnostic_logic import DiagnosticManager
from learner_store import build_learner_snapshot, learner_exists, load_learner, save_learner
from retriever import retrieve_context_bundle

llm = ChatNVIDIA(
  model="moonshotai/kimi-k2.5",
  api_key="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",
  temperature=1,
  top_p=1,
  max_completion_tokens=16384,
)

LEVEL_GUIDELINES = {
    "A1": "Use short sentences, simple vocabulary, and one main grammar point at a time.",
    "A2": "Use clear explanations with one or two linked ideas and everyday examples.",
    "B1": "Use fuller explanations, contrast patterns when helpful, and include one extension tip.",
}


def default_learner_profile():
    return {
        "current_goal": "",
        "recent_topics": [],
        "recent_grammar_points": [],
        "weak_topics": [],
        "strong_topics": [],
        "preferred_language_support": "mostly_english",
        "last_goal_type": "",
    }


def default_grammar_point_mastery():
    return {}


class TutorState(TypedDict, total=False):
    phase: str
    user_level: str
    messages: List[Dict[str, str]]
    diagnostic_id: int
    diagnostic_results: Dict[int, int]
    diagnostic_feedback: List[Dict[str, Any]]
    intro_shown: bool

    latest_user_message: str
    topic_hint: str
    grammar_point: str
    retrieved_context: str
    retrieved_documents: List[Dict[str, str]]
    retrieval_used_fallback: bool
    lesson_plan: Dict[str, Any]

    goal_type: str
    response_style: str
    language_support: str
    practice_now: str
    routing_rationale: str

    draft_response: str
    quality_status: str
    quality_rationale: str

    learner_profile: Dict[str, Any]
    grammar_point_mastery: Dict[str, int]

    learner_id: str
    display_name: str
    is_returning_learner: bool
    wants_retake_diagnostic: bool

    level_source: str
    level_confidence: str
    level_change_intent: str
    requested_level: str
    level_change_rationale: str


def get_latest_user_message(messages):
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"].strip()
    return ""


def extract_section(text, field_name):
    prefix = f"{field_name}:"
    for line in text.splitlines():
        if line.upper().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def build_diagnostic_intro():
    return (
        "Hallo! I’m Lexi, your German tutor. "
        "I’ll ask a few short questions first so I can understand your current level and teach in a way that fits you. "
        "Please answer in German as naturally as you can. If you're unsure, just try your best.\n\n"
        "Let’s begin."
    )


def classify_request_dimensions(user_message, user_level="A1"):
    prompt = f"""
You are routing a learner message for a German tutor.

Student level: {user_level}
User message: {user_message}

Classify the message along these dimensions:

GOAL_TYPE:
- explanation
- practice
- correction
- study_plan
- general_help

RESPONSE_STYLE:
- gentle
- structured
- brief

LANGUAGE_SUPPORT:
- mostly_english
- mixed
- mostly_german

PRACTICE_NOW:
- YES
- NO

Rules:
- If the learner asks for a plan, roadmap, or schedule, GOAL_TYPE should usually be study_plan.
- If the learner asks for a plan, PRACTICE_NOW should usually be NO unless they explicitly ask to start now.
- For A1 learners, prefer mostly_english unless the user clearly asks for more German.
- For A2 learners, mixed is usually appropriate.
- For B1 learners, mixed or mostly_german can be appropriate depending on the request.

Reply exactly in this format:
GOAL_TYPE: one label
RESPONSE_STYLE: one label
LANGUAGE_SUPPORT: one label
PRACTICE_NOW: YES or NO
RATIONALE: one short sentence
""".strip()

    response = llm.invoke(prompt).content.strip()

    goal_type = extract_section(response, "GOAL_TYPE") or "general_help"
    response_style = extract_section(response, "RESPONSE_STYLE") or "gentle"
    language_support = extract_section(response, "LANGUAGE_SUPPORT") or "mostly_english"
    practice_now = extract_section(response, "PRACTICE_NOW").upper() or "NO"
    rationale = extract_section(response, "RATIONALE") or "Routing based on learner request."

    if goal_type not in {"explanation", "practice", "correction", "study_plan", "general_help"}:
        goal_type = "general_help"
    if response_style not in {"gentle", "structured", "brief"}:
        response_style = "gentle"
    if language_support not in {"mostly_english", "mixed", "mostly_german"}:
        language_support = "mostly_english"
    if practice_now not in {"YES", "NO"}:
        practice_now = "NO"

    return {
        "goal_type": goal_type,
        "response_style": response_style,
        "language_support": language_support,
        "practice_now": practice_now,
        "routing_rationale": rationale,
    }


def _fallback_level_adjustment_request(user_message: str, current_level: str):
    lowered = user_message.lower()
    match = re.search(r"\b(a1|a2|b1)\b", lowered)

    direct_markers = [
        "i am",
        "i'm",
        "set my level",
        "change my level",
        "not ",
        "instead of",
        "actually",
    ]
    difficulty_markers = [
        "too easy",
        "too hard",
        "difficult",
        "this level is wrong",
        "not my level",
    ]

    if match:
        requested = match.group(1).upper()
        is_direct = any(marker in lowered for marker in direct_markers) or f"not {current_level.lower()}" in lowered
        return {
            "level_change_intent": "YES",
            "requested_level": requested,
            "level_confidence": "high" if is_direct else "medium",
            "level_change_rationale": "Fallback parser detected explicit learner level mention.",
        }

    if any(marker in lowered for marker in difficulty_markers):
        return {
            "level_change_intent": "YES",
            "requested_level": "NONE",
            "level_confidence": "medium",
            "level_change_rationale": "Fallback parser detected level-difficulty mismatch signal.",
        }

    return {
        "level_change_intent": "NO",
        "requested_level": "NONE",
        "level_confidence": "low",
        "level_change_rationale": "Fallback parser found no level-change intent.",
    }


def classify_level_adjustment_request(user_message, current_level):
    prompt = f"""
You are checking whether a learner wants to change their German level.

Current level: {current_level}
User message: {user_message}

Decide:
- LEVEL_CHANGE_INTENT: YES or NO
- REQUESTED_LEVEL: A1, A2, B1, or NONE
- CONFIDENCE: HIGH, MEDIUM, or LOW

Rules:
- Use YES only if the learner is clearly discussing their level or asking for a level change.
- REQUESTED_LEVEL should be one of A1, A2, B1 only if explicitly stated or strongly implied.
- If the learner only says the material is too easy or too hard, LEVEL_CHANGE_INTENT may be YES but REQUESTED_LEVEL can be NONE.
- If the learner explicitly says things like "I am A2", "I think I am B1", "set my level to A2", or "I am not A1", that counts as a real level-change request.

Reply exactly in this format:
LEVEL_CHANGE_INTENT: YES or NO
REQUESTED_LEVEL: A1 or A2 or B1 or NONE
CONFIDENCE: HIGH or MEDIUM or LOW
RATIONALE: one short sentence
""".strip()

    try:
        response = llm.invoke(prompt).content.strip()

        intent = extract_section(response, "LEVEL_CHANGE_INTENT").upper() or "NO"
        requested_level = extract_section(response, "REQUESTED_LEVEL").upper() or "NONE"
        confidence = extract_section(response, "CONFIDENCE").upper() or "LOW"
        rationale = extract_section(response, "RATIONALE") or "No level change detected."

        if intent not in {"YES", "NO"}:
            intent = "NO"
        if requested_level not in {"A1", "A2", "B1", "NONE"}:
            requested_level = "NONE"
        if confidence not in {"HIGH", "MEDIUM", "LOW"}:
            confidence = "LOW"

        return {
            "level_change_intent": intent,
            "requested_level": requested_level,
            "level_confidence": confidence.lower(),
            "level_change_rationale": rationale,
        }

    except Exception:
        return _fallback_level_adjustment_request(user_message, current_level)


def detect_topic(user_message):
    lowered = user_message.lower()
    topic_keywords = {
        "Articles": ["article", "artikel", "der", "die", "das", "ein", "einen", "den", "dem"],
        "Negation": ["negation", "kein", "keine", "nicht", "verneinung"],
        "Verb Conjugation": ["verb", "conjugation", "konjugation", "perfekt", "past tense"],
        "Cases": ["case", "akkusativ", "dativ", "nominativ", "genitiv", "preposition"],
        "Sentence Structure": ["word order", "sentence structure", "stellung", "weil", "nebensatz", "relative clause"],
        "Grammar": ["grammar", "grammatik", "konjunktiv", "comparative", "vergleich"],
    }
    for topic, keywords in topic_keywords.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "Grammar"


def grade_diagnostic_answer(task, user_answer):
    prompt = f"""
You are grading a German placement-test answer.

Level: {task['level']}
Topic: {task['topic']}
Grammar point: {task['grammar_point']}
Diagnostic goal: {task['prompt_goal']}
Criteria: {task['criteria']}
Reference answer: {task['example_answer']}
Student answer: {user_answer}

Grade with these labels:
- FULL: clearly demonstrates the target grammar point
- PARTIAL: partially demonstrates it, but with weaknesses or incompleteness
- FAIL: incorrect, avoids the target grammar, or is too weak to count

Be strict but fair.
Minor spelling mistakes are acceptable if the grammar target is still clear.

Reply exactly in this format:
SCORE: FULL or PARTIAL or FAIL
RATIONALE: one short sentence
""".strip()

    response = llm.invoke(prompt).content.strip()
    score_label = extract_section(response, "SCORE").upper()
    rationale = extract_section(response, "RATIONALE") or "I checked the answer against the target grammar."

    score_map = {
        "FULL": 2,
        "PARTIAL": 1,
        "FAIL": 0,
    }
    score_value = score_map.get(score_label, 0)

    return {
        "score_label": score_label if score_label in score_map else "FAIL",
        "score_value": score_value,
        "correct": score_value >= 1,
        "rationale": rationale,
    }


def generate_diagnostic_question(task, user_level="A1"):
    prompt = f"""
You are Lexi, a warm German tutor creating one short level-check question.

Target learner band: {task['level']}
Current learner estimate: {user_level}
Topic: {task['topic']}
Grammar point: {task['grammar_point']}
Diagnostic goal: {task['prompt_goal']}

Write one short question or instruction that tests this grammar point.

Rules:
- The learner should answer in German.
- Keep the wording natural and teacher-like.
- Use English when giving the instruction.
- Do not include the answer.
- Keep it short.
- Do not label difficulty or mention CEFR.

Reply with only the question text.
""".strip()

    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return f"Please answer in German: {task['prompt_goal']}"


def build_human_diagnostic_feedback(task, evaluation, user_level="A1"):
    prompt = f"""
You are Lexi, a warm and supportive German tutor.

Student level estimate: {user_level}
Task topic: {task['topic']}
Grammar point: {task['grammar_point']}
Evaluation score: {evaluation['score_label']}
Evaluation rationale: {evaluation['rationale']}

Write a short tutor response after the learner answers a level-check question.

Rules:
- Sound human, warm, and supportive.
- Keep it short: 1 to 2 sentences.
- If FULL, acknowledge it naturally.
- If PARTIAL, be encouraging and signal that the learner is on the right track.
- If FAIL, be gentle and reassuring.
- Do not over-explain yet.
- Do not say "diagnostic", "verdict", "yes", or "no".

Reply with only the tutor message.
""".strip()

    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        if evaluation["score_value"] == 2:
            return "Nice work. That was a strong answer."
        if evaluation["score_value"] == 1:
            return "Good start. You're on the right track."
        return "Good try. Let's keep going one step at a time."


def ask_diagnostic_question(state):
    current_id = state.get("diagnostic_id", DiagnosticManager.get_start_task_id())
    task = DiagnosticManager.get_task(current_id)
    generated_question = generate_diagnostic_question(
        task,
        user_level=state.get("user_level", "A1"),
    )
    question_prompt = DiagnosticManager.format_question(task, generated_question)

    new_messages = list(state.get("messages", []))

    if not state.get("intro_shown", False):
        intro_message = build_diagnostic_intro()
        new_messages.append({"role": "assistant", "content": f"{intro_message}\n\n{question_prompt}"})
        return {
            "phase": "diagnostic",
            "diagnostic_id": current_id,
            "intro_shown": True,
            "messages": new_messages,
        }

    new_messages.append({"role": "assistant", "content": question_prompt})
    return {
        "phase": "diagnostic",
        "diagnostic_id": current_id,
        "intro_shown": True,
        "messages": new_messages,
    }


def run_diagnostic(state):
    messages = list(state.get("messages", []))
    results = dict(state.get("diagnostic_results", {}))
    feedback_history = list(state.get("diagnostic_feedback", []))

    if not messages or messages[-1]["role"] != "user":
        return ask_diagnostic_question(state)

    current_id = state.get("diagnostic_id", DiagnosticManager.get_start_task_id())
    task = DiagnosticManager.get_task(current_id)
    evaluation = grade_diagnostic_answer(task, messages[-1]["content"])
    results[current_id] = evaluation["score_value"]

    feedback_history.append(
        {
            "task_id": current_id,
            "level": task["level"],
            "topic": task["topic"],
            "grammar_point": task["grammar_point"],
            "score_label": evaluation["score_label"],
            "score_value": evaluation["score_value"],
            "correct": evaluation["score_value"] >= 1,
            "rationale": evaluation["rationale"],
        }
    )

    feedback = build_human_diagnostic_feedback(
        task=task,
        evaluation=evaluation,
        user_level=state.get("user_level", "A1"),
    )

    next_id = DiagnosticManager.get_next_task_id(current_id, evaluation["score_value"], results)

    if next_id is None:
        final_level = DiagnosticManager.determine_final_level(results)
        completion_message = DiagnosticManager.build_completion_message(final_level, results)
        assistant_message = f"{feedback}\n\n{completion_message}"
        learner_profile = build_learner_profile_from_diagnostic(
            feedback_history,
            state.get("learner_profile", default_learner_profile()),
        )
        grammar_point_mastery = update_mastery_from_diagnostic(
            state.get("grammar_point_mastery", default_grammar_point_mastery()),
            feedback_history,
        )

        updated_state = {
            "phase": "tutoring",
            "user_level": final_level,
            "diagnostic_id": 0,
            "diagnostic_results": results,
            "diagnostic_feedback": feedback_history,
            "learner_profile": learner_profile,
            "grammar_point_mastery": grammar_point_mastery,
            "intro_shown": True,
            "messages": messages + [{"role": "assistant", "content": assistant_message}],
            "latest_user_message": "",
            "grammar_point": "",
            "level_source": "diagnostic",
            "level_confidence": "high",
        }

        learner_id = state.get("learner_id", "").strip()
        if learner_id:
            snapshot_source = dict(state)
            snapshot_source.update(updated_state)
            snapshot = build_learner_snapshot(snapshot_source)
            snapshot["display_name"] = state.get("display_name", learner_id)
            save_learner(learner_id, snapshot)

        return updated_state

    next_task = DiagnosticManager.get_task(next_id)
    next_generated_question = generate_diagnostic_question(
        next_task,
        user_level=state.get("user_level", "A1"),
    )
    next_prompt = DiagnosticManager.format_question(next_task, next_generated_question)
    assistant_message = f"{feedback}\n\n{next_prompt}"

    return {
        "phase": "diagnostic",
        "diagnostic_id": next_id,
        "diagnostic_results": results,
        "diagnostic_feedback": feedback_history,
        "intro_shown": True,
        "messages": messages + [{"role": "assistant", "content": assistant_message}],
    }


def analyze_query(state):
    latest_user_message = get_latest_user_message(state.get("messages", []))
    user_level = state.get("user_level", "A1")

    level_change = classify_level_adjustment_request(latest_user_message, user_level)
    routing = classify_request_dimensions(latest_user_message, user_level=user_level)

    updates = {
        "latest_user_message": latest_user_message,
        "goal_type": routing["goal_type"],
        "response_style": routing["response_style"],
        "language_support": routing["language_support"],
        "practice_now": routing["practice_now"],
        "routing_rationale": routing["routing_rationale"],
        "topic_hint": detect_topic(latest_user_message),
        "level_change_intent": level_change["level_change_intent"],
        "requested_level": level_change["requested_level"],
        "level_change_rationale": level_change["level_change_rationale"],
    }

    if level_change["level_change_intent"] == "YES" and level_change["requested_level"] in {"A1", "A2", "B1"}:
        updates["user_level"] = level_change["requested_level"]
        updates["level_source"] = "learner_override"
        updates["level_confidence"] = level_change["level_confidence"]
        updates["routing_rationale"] = (
            f"{routing['routing_rationale']} Level override requested: "
            f"{level_change['requested_level']}. {level_change['level_change_rationale']}"
        )

    return updates


def retrieve_context(state):
    goal_type = state.get("goal_type", "general_help")
    topic_hint = state.get("topic_hint", "Grammar")
    latest_user_message = state.get("latest_user_message", "")
    user_level = state.get("user_level", "A1")

    if goal_type == "study_plan":
        return {
            "retrieved_context": "",
            "retrieved_documents": [],
            "retrieval_used_fallback": False,
            "topic_hint": topic_hint,
            "grammar_point": "",
        }

    bundle = retrieve_context_bundle(
        query=latest_user_message,
        user_level=user_level,
        topic_hint=topic_hint,
        k=4,
    )
    return {
        "topic_hint": bundle["topic"],
        "grammar_point": bundle.get("grammar_point", ""),
        "retrieved_context": bundle["context_text"],
        "retrieved_documents": bundle["documents"],
        "retrieval_used_fallback": bundle["used_fallback"],
    }


def plan_lesson(state):
    level = state.get("user_level", "A1")
    goal_type = state.get("goal_type", "general_help")
    response_style = state.get("response_style", "gentle")
    language_support = state.get("language_support", "mostly_english")
    practice_now = state.get("practice_now", "NO")

    return {
        "lesson_plan": {
            "level_guideline": LEVEL_GUIDELINES.get(level, LEVEL_GUIDELINES["A1"]),
            "goal_type": goal_type,
            "response_style": response_style,
            "language_support": language_support,
            "practice_now": practice_now,
            "topic": state.get("topic_hint", "Grammar"),
            "grammar_point": state.get("grammar_point", ""),
            "use_retrieval_fallback": state.get("retrieval_used_fallback", False),
        }
    }


def get_feedback_score_value(item):
    if "score_value" in item:
        return item["score_value"]
    if "correct" in item:
        return 1 if item["correct"] else 0
    return 0


def summarize_diagnostic_feedback(feedback_history):
    if not feedback_history:
        return "No diagnostic feedback is available."

    strengths = []
    weaknesses = []

    for item in feedback_history:
        label = f"{item.get('level', 'Unknown')} {item.get('topic', 'Grammar')}"
        if get_feedback_score_value(item) >= 1:
            strengths.append(label)
        else:
            weaknesses.append(label)

    strengths = list(dict.fromkeys(strengths))
    weaknesses = list(dict.fromkeys(weaknesses))

    strength_text = ", ".join(strengths) if strengths else "none identified yet"
    weakness_text = ", ".join(weaknesses) if weaknesses else "none identified yet"

    return f"Strengths: {strength_text}. Weaknesses: {weakness_text}."


def unique_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def build_learner_profile_from_diagnostic(feedback_history, current_profile=None):
    profile = dict(current_profile or default_learner_profile())

    weak_topics = list(profile.get("weak_topics", []))
    strong_topics = list(profile.get("strong_topics", []))

    for item in feedback_history:
        topic_label = f"{item.get('level', 'Unknown')} {item.get('topic', 'Grammar')}"
        if get_feedback_score_value(item) >= 1:
            strong_topics.append(topic_label)
        else:
            weak_topics.append(topic_label)

    profile["weak_topics"] = unique_keep_order(weak_topics)[-6:]
    profile["strong_topics"] = unique_keep_order(strong_topics)[-6:]
    return profile


def update_learner_profile(profile, state):
    profile = dict(profile or default_learner_profile())

    topic = state.get("topic_hint", "")
    grammar_point = state.get("grammar_point", "")
    goal_type = state.get("goal_type", "")
    language_support = state.get("language_support", profile.get("preferred_language_support", "mostly_english"))
    latest_user_message = state.get("latest_user_message", "").strip()

    recent_topics = list(profile.get("recent_topics", []))
    if topic:
        recent_topics.append(topic)

    recent_grammar_points = list(profile.get("recent_grammar_points", []))
    if grammar_point:
        recent_grammar_points.append(grammar_point)

    profile["recent_topics"] = unique_keep_order(recent_topics)[-5:]
    profile["recent_grammar_points"] = unique_keep_order(recent_grammar_points)[-5:]
    profile["preferred_language_support"] = language_support
    profile["last_goal_type"] = goal_type

    if goal_type == "study_plan":
        profile["current_goal"] = latest_user_message
    elif latest_user_message and not profile.get("current_goal"):
        profile["current_goal"] = latest_user_message

    return profile


def summarize_learner_profile(profile):
    if not profile:
        return "No learner profile is available."

    current_goal = profile.get("current_goal") or "not clearly set yet"
    recent_topics = ", ".join(profile.get("recent_topics", [])) or "none yet"
    recent_grammar_points = ", ".join(profile.get("recent_grammar_points", [])) or "none yet"
    weak_topics = ", ".join(profile.get("weak_topics", [])) or "none identified yet"
    strong_topics = ", ".join(profile.get("strong_topics", [])) or "none identified yet"
    preferred_language_support = profile.get("preferred_language_support", "mostly_english")
    last_goal_type = profile.get("last_goal_type") or "unknown"

    return (
        f"Current goal: {current_goal}. "
        f"Recent topics: {recent_topics}. "
        f"Recent grammar points: {recent_grammar_points}. "
        f"Weak topics: {weak_topics}. "
        f"Strong topics: {strong_topics}. "
        f"Preferred language support: {preferred_language_support}. "
        f"Last goal type: {last_goal_type}."
    )


def clamp_mastery(value):
    return max(0, min(3, value))


def update_mastery_from_diagnostic(grammar_point_mastery, feedback_history):
    mastery = dict(grammar_point_mastery or {})

    for item in feedback_history:
        grammar_point = item.get("grammar_point")
        if not grammar_point:
            continue

        current = mastery.get(grammar_point, 0)
        score_value = get_feedback_score_value(item)

        if score_value == 2:
            mastery[grammar_point] = clamp_mastery(current + 1)
        elif score_value == 1:
            mastery[grammar_point] = clamp_mastery(current)
        else:
            mastery[grammar_point] = clamp_mastery(current)

    return mastery


def update_mastery_from_session(grammar_point_mastery, state):
    mastery = dict(grammar_point_mastery or {})

    grammar_point = state.get("grammar_point", "")
    goal_type = state.get("goal_type", "")
    latest_user_message = state.get("latest_user_message", "").lower()

    if not grammar_point:
        return mastery

    current = mastery.get(grammar_point, 0)

    if goal_type in {"explanation", "study_plan", "general_help"}:
        mastery[grammar_point] = current
    elif goal_type == "practice":
        if any(word in latest_user_message for word in ["easy", "understand", "got it", "i know", "clear"]):
            mastery[grammar_point] = clamp_mastery(current + 1)
        elif any(word in latest_user_message for word in ["confused", "hard", "difficult", "don't understand", "not clear"]):
            mastery[grammar_point] = clamp_mastery(current - 1)
        else:
            mastery[grammar_point] = current
    elif goal_type == "correction":
        mastery[grammar_point] = clamp_mastery(current)

    return mastery


def summarize_grammar_point_mastery(grammar_point_mastery):
    if not grammar_point_mastery:
        return "No grammar-point mastery data is available yet."

    weak = [gp for gp, score in grammar_point_mastery.items() if score <= 1]
    strong = [gp for gp, score in grammar_point_mastery.items() if score >= 2]

    weak_text = ", ".join(weak) if weak else "none yet"
    strong_text = ", ".join(strong) if strong else "none yet"

    return f"Weaker grammar points: {weak_text}. Stronger grammar points: {strong_text}."


def build_language_support_instructions(language_support):
    if language_support == "mostly_english":
        return (
            "Use mostly English for explanations. "
            "Use only short German examples. "
            "Always translate German examples into English."
        )
    if language_support == "mixed":
        return (
            "Use a balanced mix of English explanation and short German examples. "
            "Translate or gloss important German phrases."
        )
    return (
        "You may use more German, but keep the explanation understandable. "
        "Add English support when the learner may struggle."
    )


def build_shared_tutor_instructions(state):
    level = state.get("user_level", "A1")
    lesson_plan = state.get("lesson_plan", {})
    diagnostic_summary = summarize_diagnostic_feedback(state.get("diagnostic_feedback", []))
    learner_profile_summary = summarize_learner_profile(
        state.get("learner_profile", default_learner_profile())
    )
    mastery_summary = summarize_grammar_point_mastery(
        state.get("grammar_point_mastery", default_grammar_point_mastery())
    )
    response_style = lesson_plan.get("response_style", "gentle")
    language_support = lesson_plan.get("language_support", "mostly_english")
    grammar_point = lesson_plan.get("grammar_point", "") or state.get("grammar_point", "")
    level_source = state.get("level_source", "diagnostic")
    level_confidence = state.get("level_confidence", "high")

    style_instruction_map = {
        "gentle": "Use a warm, encouraging, human tone.",
        "structured": "Be very clear and well-organized.",
        "brief": "Keep the answer concise but still supportive.",
    }

    return f"""
You are Lexi, a warm, smart, adaptive German tutor.

Student level: {level}
Level source: {level_source}
Level confidence: {level_confidence}
Topic: {lesson_plan.get('topic', 'Grammar')}
Grammar point: {grammar_point or 'not clearly identified'}
Teaching style: {lesson_plan.get('level_guideline', LEVEL_GUIDELINES['A1'])}
Response style: {response_style}
Language support: {language_support}

Diagnostic profile:
{diagnostic_summary}

Learner profile:
{learner_profile_summary}

Grammar-point mastery:
{mastery_summary}

Routing rationale:
{state.get('routing_rationale', 'No routing rationale available.')}

Tutor behavior rules:
- Be human, warm, and supportive.
- Do not sound like a textbook unless the learner explicitly wants that.
- For A1 learners, reduce cognitive load and avoid long German-only passages.
- Use retrieved teaching material when helpful.
- Use rule-like content for explanation and example-like content for illustration.
- Reuse the learner profile naturally, especially recent struggles and current goals.
- Use grammar-point mastery to decide whether to explain more slowly, review, or move faster.
- If grammar-point mastery is low, explain more gently and include more support.
- If grammar-point mastery is high, avoid over-explaining and move more efficiently.
- If the learner explicitly changed their level, respect that and adapt accordingly.
- If retrieval is thin, answer carefully from general knowledge.

Style instruction:
{style_instruction_map.get(response_style, style_instruction_map['gentle'])}

Language instruction:
{build_language_support_instructions(language_support)}
""".strip()


def build_branch_response(state, branch_goal, branch_structure, include_context=True):
    context_block = state.get("retrieved_context", "") or "No retrieval context was found."
    system_prompt = f"""
{build_shared_tutor_instructions(state)}

Goal:
{branch_goal}

Response structure:
{branch_structure}
""".strip()

    if include_context:
        user_prompt = f"""
Student request: {state.get('latest_user_message', '')}

Retrieved context:
{context_block}
""".strip()
    else:
        user_prompt = f"Student request: {state.get('latest_user_message', '')}"

    response = llm.invoke(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )
    return {"draft_response": response.content}


def handle_level_adjustment(state):
    requested_level = state.get("requested_level", "NONE")
    current_level = state.get("user_level", "A1")
    confidence = state.get("level_confidence", "medium")
    rationale = state.get("level_change_rationale", "The learner requested a level change.")
    display_name = state.get("display_name", "there")

    if requested_level not in {"A1", "A2", "B1"}:
        prompt = f"""
You are Lexi, a warm German tutor.

Current learner level: {current_level}
Learner message: {state.get('latest_user_message', '')}
Reason detected: {rationale}

Write a short response that:
- acknowledges the learner's concern about difficulty or level
- sounds human and supportive
- offers either a quick adjustment or a retake of the level check
- does not force a level change
- stays concise

Reply with only the tutor message.
""".strip()

        response = llm.invoke(prompt).content.strip()
        return {
            "draft_response": response,
        }

    prompt = f"""
You are Lexi, a warm German tutor.

Learner name: {display_name}
Current level: {current_level}
Requested level: {requested_level}
Confidence: {confidence}
Reason detected: {rationale}

Write a short response that:
- acknowledges the learner politely
- confirms the level will be adjusted
- sounds human and supportive
- says the tutoring will adapt from now on
- stays concise

Reply with only the tutor message.
""".strip()

    response = llm.invoke(prompt).content.strip()
    return {
        "user_level": requested_level,
        "level_source": "learner_override",
        "level_confidence": confidence,
        "draft_response": response,
    }


def study_plan_node(state):
    return build_branch_response(
        state,
        branch_goal=(
            "Create a practical study plan. "
            "Do not start an exercise automatically. "
            "Use mostly English for A1 unless the learner asks otherwise."
        ),
        branch_structure=(
            "1. one warm opening sentence\n"
            "2. a clear study plan\n"
            "3. short daily or step-based guidance\n"
            "4. at most one or two tiny German examples with English support\n"
            "5. a gentle offer for the next step"
        ),
        include_context=False,
    )


def explain_concept_node(state):
    return build_branch_response(
        state,
        branch_goal="Explain the concept clearly first. Do not force practice unless it feels natural.",
        branch_structure=(
            "1. a short explanation\n"
            "2. one German example with English gloss\n"
            "3. one short natural next step only if helpful"
        ),
        include_context=True,
    )


def run_practice_node(state):
    return build_branch_response(
        state,
        branch_goal=(
            "Run a short guided practice activity. "
            "Teach briefly first, then give a small exercise. "
            "For A1, keep the exercise very easy and well-supported."
        ),
        branch_structure=(
            "1. a short explanation\n"
            "2. one example with English gloss\n"
            "3. one short guided exercise\n"
            "4. encouragement"
        ),
        include_context=True,
    )


def correct_answer_node(state):
    return build_branch_response(
        state,
        branch_goal="Correct the learner gently. Explain what changed and why. Keep the tone reassuring.",
        branch_structure=(
            "1. a gentle correction\n"
            "2. a short explanation\n"
            "3. one improved example with English gloss\n"
            "4. one optional follow-up prompt"
        ),
        include_context=True,
    )


def general_help_node(state):
    return build_branch_response(
        state,
        branch_goal="Answer helpfully and naturally without forcing a rigid format.",
        branch_structure=(
            "1. a helpful direct response\n"
            "2. one example if useful\n"
            "3. a gentle next step only if it helps"
        ),
        include_context=True,
    )


def response_quality_check(state):
    draft_response = state.get("draft_response", "")
    level = state.get("user_level", "A1")
    goal_type = state.get("goal_type", "general_help")
    language_support = state.get("language_support", "mostly_english")
    practice_now = state.get("practice_now", "NO")

    prompt = f"""
You are reviewing a German tutor response before it is shown to the learner.

Student level: {level}
Goal type: {goal_type}
Language support: {language_support}
Practice now: {practice_now}

Draft response:
{draft_response}

Check whether the draft is suitable.

Important checks:
- For A1, avoid long German-only passages.
- If language support is mostly_english, explanations should mainly be in English.
- If the learner asked for a study plan, do not suddenly start a quiz unless practice_now is YES.
- The tone should feel warm and human.
- The answer should match the learner's request.

Reply exactly in this format:
STATUS: PASS or REVISE
RATIONALE: one short sentence
""".strip()

    response = llm.invoke(prompt).content.strip()
    status = extract_section(response, "STATUS").upper() or "PASS"
    rationale = extract_section(response, "RATIONALE") or "The draft looks suitable."

    if status not in {"PASS", "REVISE"}:
        status = "PASS"

    return {
        "quality_status": status,
        "quality_rationale": rationale,
    }


def answer_revision(state):
    draft_response = state.get("draft_response", "")
    quality_rationale = state.get("quality_rationale", "Please improve the response.")
    level = state.get("user_level", "A1")
    goal_type = state.get("goal_type", "general_help")
    language_support = state.get("language_support", "mostly_english")

    prompt = f"""
You are revising a tutor response for a German learner.

Student level: {level}
Goal type: {goal_type}
Language support: {language_support}

Original draft:
{draft_response}

Reviewer feedback:
{quality_rationale}

Revise the response so it:
- fits the learner's level
- sounds warm and human
- uses enough English support for the learner
- matches the learner's actual request
- does not start practice unexpectedly

Reply with only the improved tutor response.
""".strip()

    response = llm.invoke(prompt).content.strip()
    return {"draft_response": response}


def finalize_response(state):
    draft_response = state.get("draft_response", "")
    return {
        "messages": state.get("messages", []) + [{"role": "assistant", "content": draft_response}]
    }


def session_memory_update(state):
    profile = update_learner_profile(
        state.get("learner_profile", default_learner_profile()),
        state,
    )
    grammar_point_mastery = update_mastery_from_session(
        state.get("grammar_point_mastery", default_grammar_point_mastery()),
        state,
    )

    updated = {
        "learner_profile": profile,
        "grammar_point_mastery": grammar_point_mastery,
    }

    learner_id = state.get("learner_id", "").strip()
    if learner_id:
        snapshot_source = dict(state)
        snapshot_source.update(updated)
        snapshot = build_learner_snapshot(snapshot_source)
        snapshot["display_name"] = state.get("display_name", learner_id)
        save_learner(learner_id, snapshot)

    return updated


def route_from_start(state):
    return state.get("phase", "diagnostic")


def route_after_plan(state):
    if (
        state.get("level_change_intent") == "YES"
        and state.get("requested_level") in {"A1", "A2", "B1", "NONE"}
    ):
        return "level_adjustment"

    goal_type = state.get("goal_type", "general_help")
    if goal_type == "study_plan":
        return "study_plan"
    if goal_type == "practice":
        return "practice"
    if goal_type == "correction":
        return "correction"
    if goal_type == "explanation":
        return "explanation"
    return "general_help"


def route_quality(state):
    return "revise" if state.get("quality_status") == "REVISE" else "finalize"


workflow = StateGraph(TutorState)
workflow.add_node("diagnostic", run_diagnostic)
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("plan_lesson", plan_lesson)

workflow.add_node("handle_level_adjustment", handle_level_adjustment)
workflow.add_node("study_plan", study_plan_node)
workflow.add_node("explain_concept", explain_concept_node)
workflow.add_node("run_practice", run_practice_node)
workflow.add_node("correct_answer", correct_answer_node)
workflow.add_node("general_help", general_help_node)

workflow.add_node("response_quality_check", response_quality_check)
workflow.add_node("answer_revision", answer_revision)
workflow.add_node("finalize_response", finalize_response)
workflow.add_node("session_memory_update", session_memory_update)

workflow.add_conditional_edges(
    START,
    route_from_start,
    {
        "diagnostic": "diagnostic",
        "tutoring": "analyze_query",
    },
)

workflow.add_edge("diagnostic", END)
workflow.add_edge("analyze_query", "retrieve_context")
workflow.add_edge("retrieve_context", "plan_lesson")

workflow.add_conditional_edges(
    "plan_lesson",
    route_after_plan,
    {
        "level_adjustment": "handle_level_adjustment",
        "study_plan": "study_plan",
        "practice": "run_practice",
        "correction": "correct_answer",
        "explanation": "explain_concept",
        "general_help": "general_help",
    },
)

workflow.add_edge("handle_level_adjustment", "response_quality_check")
workflow.add_edge("study_plan", "response_quality_check")
workflow.add_edge("explain_concept", "response_quality_check")
workflow.add_edge("run_practice", "response_quality_check")
workflow.add_edge("correct_answer", "response_quality_check")
workflow.add_edge("general_help", "response_quality_check")

workflow.add_conditional_edges(
    "response_quality_check",
    route_quality,
    {
        "revise": "answer_revision",
        "finalize": "finalize_response",
    },
)

workflow.add_edge("answer_revision", "finalize_response")
workflow.add_edge("finalize_response", "session_memory_update")
workflow.add_edge("session_memory_update", END)

app = workflow.compile()


def build_initial_state(learner_id: str, display_name: str) -> Dict[str, Any]:
    return {
        "phase": "diagnostic",
        "messages": [],
        "user_level": "Unknown",
        "diagnostic_id": DiagnosticManager.get_start_task_id(),
        "diagnostic_results": {},
        "diagnostic_feedback": [],
        "intro_shown": False,
        "learner_profile": default_learner_profile(),
        "grammar_point_mastery": default_grammar_point_mastery(),
        "learner_id": learner_id,
        "display_name": display_name,
        "is_returning_learner": False,
        "wants_retake_diagnostic": False,
        "level_source": "unknown",
        "level_confidence": "low",
        "level_change_intent": "NO",
        "requested_level": "NONE",
        "level_change_rationale": "",
    }


def build_state_from_saved_learner(learner_id: str, saved: Dict[str, Any]) -> Dict[str, Any]:
    display_name = saved.get("display_name", learner_id)
    level = saved.get("user_level", "A1")

    return {
        "phase": "tutoring",
        "messages": [],
        "user_level": level,
        "diagnostic_id": 0,
        "diagnostic_results": saved.get("diagnostic_results", {}),
        "diagnostic_feedback": saved.get("diagnostic_feedback", []),
        "intro_shown": True,
        "learner_profile": saved.get("learner_profile", default_learner_profile()),
        "grammar_point_mastery": saved.get("grammar_point_mastery", default_grammar_point_mastery()),
        "learner_id": learner_id,
        "display_name": display_name,
        "is_returning_learner": True,
        "wants_retake_diagnostic": False,
        "level_source": saved.get("level_source", "diagnostic"),
        "level_confidence": saved.get("level_confidence", "medium"),
        "level_change_intent": "NO",
        "requested_level": "NONE",
        "level_change_rationale": "",
    }


if __name__ == "__main__":
    learner_name = input("Learner name: ").strip()
    while not learner_name:
        learner_name = input("Learner name: ").strip()

    learner_id = learner_name.lower()

    if learner_exists(learner_id):
        saved = load_learner(learner_id) or {}
        saved_level = saved.get("user_level", "Unknown")
        choice = input(
            f"Welcome back, {saved.get('display_name', learner_name)}. "
            f"I remember you around {saved_level}. Type 'continue' to resume or 'retake' to do the level check again: "
        ).strip().lower()

        while choice not in {"continue", "retake"}:
            choice = input("Type 'continue' or 'retake': ").strip().lower()

        if choice == "continue":
            current_state = build_state_from_saved_learner(learner_id, saved)
            welcome_back = (
                f"Welcome back, {saved.get('display_name', learner_name)}. "
                f"We'll continue from your current level, {saved_level}. "
                "What would you like to work on today?"
            )
            current_state["messages"].append({"role": "assistant", "content": welcome_back})
            print(f"\nLEXI: {current_state['messages'][-1]['content']}")
        else:
            current_state = build_initial_state(learner_id, learner_name)
            current_state["is_returning_learner"] = True
            current_state["wants_retake_diagnostic"] = True
            current_state = app.invoke(current_state)
            print(f"\nLEXI: {current_state['messages'][-1]['content']}")
    else:
        current_state = build_initial_state(learner_id, learner_name)
        current_state = app.invoke(current_state)
        print(f"\nLEXI: {current_state['messages'][-1]['content']}")

    while True:
        user_text = input("YOU: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            if current_state.get("learner_id"):
                snapshot = build_learner_snapshot(current_state)
                snapshot["display_name"] = current_state.get("display_name", learner_name)
                save_learner(current_state["learner_id"], snapshot)
            print("Session ended.")
            break
        if not user_text:
            continue

        current_state["messages"].append({"role": "user", "content": user_text})
        current_state = app.invoke(current_state)
        print(f"\nLEXI: {current_state['messages'][-1]['content']}")
