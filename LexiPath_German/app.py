import re
import random
import time
from types import SimpleNamespace
from typing import Any, Dict, List, TypedDict

from openai import OpenAI
from langgraph.graph import END, START, StateGraph

from config import (
    MOONSHOT_BASE_URL,
    MOONSHOT_MAX_TOKENS,
    MOONSHOT_MODEL,
    MOONSHOT_TEMPERATURE,
    MOONSHOT_TIMEOUT_SECONDS,
    MOONSHOT_TOP_P,
    get_moonshot_api_key,
)
from curriculum import format_syllabus_reference, select_syllabus_reference
from diagnostic_logic import DiagnosticManager
from learner_store import build_learner_snapshot, learner_exists, load_learner, save_learner
from retriever import retrieve_context_bundle


class OpenAICompatibleChat:
    def __init__(self):
        self.client = OpenAI(
            base_url=MOONSHOT_BASE_URL,
            api_key=get_moonshot_api_key(),
            timeout=MOONSHOT_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def invoke(self, prompt_or_messages):
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages

        completion = self.client.chat.completions.create(
            model=MOONSHOT_MODEL,
            messages=messages,
            temperature=MOONSHOT_TEMPERATURE,
            top_p=min(MOONSHOT_TOP_P, 0.95),
            max_tokens=MOONSHOT_MAX_TOKENS,
        )
        content = completion.choices[0].message.content or ""
        return SimpleNamespace(content=content)


llm = OpenAICompatibleChat()

LEVEL_GUIDELINES = {
    "A1": "Use short sentences, simple vocabulary, and one main grammar point at a time.",
    "A2": "Use clear explanations with one or two linked ideas and everyday examples.",
    "B1": "Use fuller explanations, contrast patterns when helpful, and include one extension tip.",
}

CEFR_LEVEL_ORDER = ["A1", "A2", "B1"]

CEFR_PROMOTION_REQUIREMENTS = {
    "A1": [
        "indefinite_articles_ein_eine_einen",
        "negation_kein",
        "present_tense_basic_verbs",
    ],
    "A2": [
        "perfect_tense_basics",
        "accusative_with_movement",
        "comparatives_basics",
    ],
    "B1": [
        "subordinate_clause_weil",
        "konjunktiv_ii_basics",
        "relative_clauses_basics",
    ],
}

CEFR_PROMOTION_MASTERY_THRESHOLD = 2


def default_learner_profile():
    return {
        "current_goal": "",
        "recent_topics": [],
        "recent_grammar_points": [],
        "weak_topics": [],
        "strong_topics": [],
        "syllabus_history": [],
        "current_syllabus_lesson": {},
        "level_progression_history": [],
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
    retrieval_focus: Dict[str, Any]
    syllabus_reference: Dict[str, Any]
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
    level_progression_status: Dict[str, Any]
    level_promoted: bool


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


TRANSIENT_LLM_ERROR_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "too many requests",
    "gateway timeout",
    "bad gateway",
)


def is_transient_llm_error(error):
    message = str(error).lower()
    return any(marker in message for marker in TRANSIENT_LLM_ERROR_MARKERS)


def invoke_llm_content(prompt_or_messages, fallback_text=None, retries=0, call_name="llm"):
    last_error = None
    started_at = time.monotonic()
    for attempt in range(retries + 1):
        try:
            content = llm.invoke(prompt_or_messages).content.strip()
            if not content:
                raise ValueError("Model returned an empty response.")
            return content
        except Exception as error:
            last_error = error
            if not is_transient_llm_error(error) or attempt >= retries:
                break
            time.sleep(2 * (attempt + 1))

    if fallback_text is not None:
        return fallback_text
    raise last_error


def fallback_request_dimensions(user_message, user_level="A1"):
    lowered = user_message.lower()
    continuation_markers = [
        "ok",
        "got it",
        "continue",
        "continue that",
        "go on",
        "next",
        "teach me",
        "you teach",
        "start",
        "let's do it",
        "lets do it",
    ]

    if any(word in lowered for word in ["practice", "exercise", "quiz", "task"]):
        goal_type = "practice"
    elif any(marker in lowered for marker in continuation_markers):
        goal_type = "practice"
    elif any(word in lowered for word in ["plan", "schedule", "roadmap", "day", "week", "lesson", "start"]):
        goal_type = "study_plan"
    elif any(word in lowered for word in ["correct", "check", "fix", "mistake"]):
        goal_type = "correction"
    elif any(word in lowered for word in ["explain", "difference", "what is", "how do", "help me understand"]):
        goal_type = "explanation"
    else:
        goal_type = "general_help"

    language_support = {
        "A1": "mostly_english",
        "A2": "mixed",
        "B1": "mixed",
    }.get(user_level, "mostly_english")

    return f"""
GOAL_TYPE: {goal_type}
RESPONSE_STYLE: gentle
LANGUAGE_SUPPORT: {language_support}
PRACTICE_NOW: {"YES" if goal_type in {"practice", "correction"} else "NO"}
RATIONALE: Fallback routing used because the model service was unavailable.
""".strip()


def build_service_fallback_response(state):
    level = state.get("user_level", "A1")
    latest_user_message = state.get("latest_user_message", "").lower()
    grammar_point = state.get("grammar_point", "")
    topic = state.get("topic_hint", "")
    profile = state.get("learner_profile", default_learner_profile())
    recent_topics = profile.get("recent_topics", [])
    recent_grammar_points = profile.get("recent_grammar_points", [])

    focus_topic = topic or (recent_topics[-1] if recent_topics else "Sentence structure")
    focus_grammar = grammar_point or (recent_grammar_points[-1] if recent_grammar_points else "general_grammar")

    if any(word in latest_user_message for word in ["plan", "schedule", "roadmap", "day", "week", "lesson", "start"]):
        return (
            "Let's keep your learning moving with a simple plan.\n\n"
            f"Today at {level}, try this:\n"
            "1. Warm-up: write 3 short German sentences about your day.\n"
            "2. Review: choose one grammar point you found difficult recently.\n"
            "3. Practice: do one small exercise with 3-5 answers.\n"
            "4. Wrap-up: correct one mistake and save one sentence to reuse tomorrow."
            "\n\n"
            "If you want, we can start with the warm-up now."
        )

    if focus_grammar == "accusative_with_movement" or focus_topic == "Basic prepositions":
        return (
            f"Let's continue at {level} with the preposition pattern we were working on.\n\n"
            "Mini-focus: use **in/auf/an + accusative** when there is movement toward a place.\n"
            "Example: *Ich gehe in die Mensa.* = I go into/to the cafeteria.\n\n"
            "Your turn: write one sentence with **in die**, **auf den**, or **an das**."
        )

    if focus_grammar == "present_tense_basic_verbs" or focus_topic == "Present tense":
        return (
            f"Let's continue at {level} with verb position in simple present sentences.\n\n"
            "Mini-focus: in a normal German main clause, the conjugated verb is in position 2.\n"
            "Example: *Am Abend schaue ich einen Film.*\n\n"
            "Your turn: write one sentence starting with **Heute** or **Am Abend**."
        )

    if focus_grammar == "konjunktiv_ii_basics" or focus_topic == "Modal verbs":
        return (
            f"Let's continue at {level} with modal-style sentence structure.\n\n"
            "Mini-focus: with modal verbs, the second verb goes to the end.\n"
            "Example: *Ich muss heute lernen.*\n\n"
            "Your turn: write one sentence with **muss**, **kann**, or **will**."
        )

    if focus_grammar == "subordinate_clause_weil":
        return (
            f"Let's continue at {level} with **weil** sentences.\n\n"
            "Mini-focus: after **weil**, the conjugated verb usually moves to the end.\n"
            "Example: *Ich lerne Deutsch, weil ich arbeiten will.*\n\n"
            "Your turn: write one sentence explaining why you study German."
        )

    return (
        f"Let's continue at {level} with the same lesson thread.\n\n"
        "Mini-focus: make one corrected sentence from your last answer and use it again naturally.\n"
        "Your turn: write one improved sentence about your day."
    )


def build_diagnostic_intro():
    return (
        "Hallo! I’m Lexi, your German tutor. It's so nice to meet you. "
        "I'll ask just a few quick questions to get a sense of your current German level and teach in a way that fits you. "
        "There's absolutely no pressure here - just share what feels comfortable, and answer in German whenever you can. Just try your best.\n\n"
        "Let’s begin."
    )


def classify_request_dimensions(user_message, user_level="A1"):
    response = fallback_request_dimensions(user_message, user_level)

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
    return _fallback_level_adjustment_request(user_message, current_level)


def detect_topic(user_message):
    lowered = user_message.lower()
    topic_keywords = {
        "Alphabet & pronunciation": ["alphabet", "pronunciation", "aussprache", "letter", "letters", "sound", "sounds", "buchstabe"],
        "Personal pronouns": ["pronoun", "pronouns", "ich", "du", "er", "sie", "wir", "ihr", "mein", "dein"],
        "Present tense": ["present tense", "verb", "conjugation", "konjugation", "gehe", "wohne", "lerne"],
        "Modal verbs": ["modal", "kann", "muss", "darf", "soll", "möchte", "würde", "konjunktiv"],
        "Sentence structure": ["word order", "sentence structure", "stellung", "weil", "nebensatz", "relative clause", "comparative", "vergleich"],
        "Articles": ["article", "artikel", "der", "die", "das", "ein", "einen", "den", "dem"],
        "Negation": ["negation", "kein", "keine", "nicht", "verneinung"],
        "Plurals": ["plural", "plurals", "mehrzahl"],
        "Basic prepositions": ["preposition", "prepositions", "case", "akkusativ", "dativ", "nominativ", "genitiv", "auf", "in", "an"],
    }
    for topic, keywords in topic_keywords.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "Sentence structure"


def _local_diagnostic_floor(task, user_answer):
    answer = (user_answer or "").strip()
    lowered = answer.lower()
    grammar_point = task.get("grammar_point", "")

    if not answer:
        return None

    full = {
        "score_label": "FULL",
        "score_value": 2,
        "correct": True,
        "rationale": "Local grammar check recognized the target structure clearly.",
    }
    partial = {
        "score_label": "PARTIAL",
        "score_value": 1,
        "correct": True,
        "rationale": "Local grammar check recognized part of the target structure.",
    }

    if grammar_point == "indefinite_articles_ein_eine_einen":
        if lowered in {"einen"}:
            return {
                **full,
                "rationale": "The answer supplies the correct masculine accusative article 'einen'.",
            }
        if re.search(r"\beinen\s+\w+", lowered):
            return {
                **full,
                "rationale": "The answer clearly uses masculine accusative 'einen' with a noun.",
            }
        if re.search(r"\bein(?:e|en)?\s+\w+", lowered):
            return partial

    if grammar_point == "negation_kein":
        if lowered in {"kein", "keine", "keinen", "keinem", "keiner", "keines"}:
            return {
                **full,
                "rationale": "The answer supplies a valid inflected form of 'kein' for the prompt.",
            }
        if re.search(r"\bkein(?:e|en|em|er|es)?\s+\w+", lowered):
            return {
                **full,
                "rationale": "The answer correctly uses an inflected form of 'kein' before a noun.",
            }
        if "kein" in lowered or "nicht" in lowered:
            return partial

    if grammar_point == "present_tense_basic_verbs":
        common_present_verbs = (
            "bin|bist|ist|sind|seid|habe|hast|hat|haben|habt|wohne|wohnst|wohnt|"
            "gehe|gehst|geht|komme|kommst|kommt|lerne|lernst|lernt|arbeite|"
            "arbeitest|arbeitet|mache|machst|macht|spiele|spielst|spielt|"
            "spreche|sprichst|spricht|esse|isst|trinke|trinkst|trinkt|"
            "lese|liest|fahren|fahre|fährst|fährt|kaufe|kaufst|kauft|"
            "brauche|brauchst|braucht|finde|findest|findet"
        )
        if re.search(rf"\b(ich|du|er|sie|es|wir|ihr)\s+({common_present_verbs})\b", lowered):
            return {
                **full,
                "rationale": "The answer is a complete present-tense sentence with a correctly conjugated verb.",
            }
        if re.search(r"\b(ich|du|er|sie|es|wir|ihr)\b", lowered) and re.search(r"\b\w+(e|st|t|en)\b", lowered):
            return partial

    if grammar_point == "perfect_tense_basics":
        has_auxiliary = re.search(r"\b(habe|hast|hat|haben|habt|bin|bist|ist|sind|seid)\b", lowered)
        has_participle = re.search(r"\bge\w+(t|en)\b|\b(gemacht|gewesen|gegangen|gefahren|gesehen|gegessen|getrunken|gelesen)\b", lowered)
        if has_auxiliary and has_participle:
            return full
        if has_auxiliary or has_participle:
            return partial

    if grammar_point == "accusative_with_movement":
        has_movement = re.search(r"\b(lege|legst|legt|stelle|stellst|stellt|hänge|hängst|hängt|gehe|gehst|geht|fahre|fährst|fährt|bringe|bringst|bringt)\b", lowered)
        has_accusative_preposition = re.search(r"\b(auf|in|an|unter|über|vor|hinter|neben|zwischen)\s+(den|die|das|einen|eine)\b", lowered)
        if has_movement and has_accusative_preposition:
            return full
        if has_accusative_preposition:
            return partial

    if grammar_point == "comparatives_basics":
        if re.search(r"\b\w+er\s+als\b", lowered):
            return full
        if "als" in lowered:
            return partial

    if grammar_point == "subordinate_clause_weil":
        weil_final_verbs = (
            "bin|bist|ist|sind|seid|habe|hast|hat|haben|habt|"
            "will|willst|wollen|muss|musst|müssen|kann|kannst|können|"
            "möchte|möchtest|möchten|lerne|lernst|lernt|arbeite|arbeitest|arbeitet|"
            "wohne|wohnst|wohnt|gehe|gehst|geht|komme|kommst|kommt|brauche|brauchst|braucht"
        )
        if "weil" in lowered and re.search(rf"\bweil\b.+\b({weil_final_verbs}|\w+(e|st|t|en))\.?$", lowered):
            return full
        if "weil" in lowered:
            return partial

    if grammar_point == "konjunktiv_ii_basics":
        if re.search(r"\b(würde|würdest|würden|hätte|hättest|hätten|wäre|wärst|wären|könnte|könntest|könnten)\b", lowered):
            return full

    if grammar_point == "relative_clauses_basics":
        if "," in lowered and re.search(r",\s*(der|die|das|den|dem|dessen|deren)\b", lowered):
            return full

    return None


def grade_diagnostic_answer(task, user_answer):
    local_floor = _local_diagnostic_floor(task, user_answer)
    if local_floor and local_floor["score_value"] == 2:
        return local_floor

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

Important grading rules:
- Grade the student's answer against the target grammar point, not against the exact reference sentence.
- The reference answer is only one possible correct answer.
- Accept different nouns, verbs, contexts, or word choices when the target grammar is clearly demonstrated.
- Accept a one-word fill-in answer if the question asks for a blank and the supplied form is correct.
- Minor spelling, capitalization, punctuation, or vocabulary mistakes are acceptable if the target grammar is still clear.
- Do not mark an answer FAIL just because it is simpler than the reference answer.
- If the target grammar is correct but the sentence has another small issue, use PARTIAL rather than FAIL.

Common examples that should be FULL:
- negation_kein: "Das ist kein Apfel." or just "kein" when filling "Das ist ___ Apfel."
- present_tense_basic_verbs: "Ich gehe jeden Tag zur Arbeit."
- indefinite_articles_ein_eine_einen: "Ich sehe einen Bruder." or just "einen" when filling an accusative masculine blank.

Reply exactly in this format:
SCORE: FULL or PARTIAL or FAIL
RATIONALE: one short sentence
""".strip()

    fallback_score = local_floor or {
        "score_label": "FAIL",
        "score_value": 0,
        "correct": False,
        "rationale": "The answer did not clearly show the target grammar in the local diagnostic check.",
    }
    fallback_response = (
        f"SCORE: {fallback_score['score_label']}\n"
        f"RATIONALE: {fallback_score['rationale']}"
    )
    response = invoke_llm_content(
        prompt,
        fallback_text=fallback_response,
        retries=0,
        call_name=f"diagnostic:grade:{task.get('grammar_point', 'unknown')}",
    )
    score_label = extract_section(response, "SCORE").upper()
    rationale = extract_section(response, "RATIONALE") or "I checked the answer against the target grammar."

    score_map = {
        "FULL": 2,
        "PARTIAL": 1,
        "FAIL": 0,
    }
    score_value = score_map.get(score_label, 0)
    llm_evaluation = {
        "score_label": score_label if score_label in score_map else "FAIL",
        "score_value": score_value,
        "correct": score_value >= 1,
        "rationale": rationale,
    }

    if local_floor and local_floor["score_value"] > llm_evaluation["score_value"]:
        local_floor["rationale"] = (
            f"{local_floor['rationale']} This overrode the LLM grader's lower score: "
            f"{llm_evaluation['score_label']}."
        )
        return local_floor

    return llm_evaluation


DIAGNOSTIC_QUESTION_VARIANTS = {
    "indefinite_articles_ein_eine_einen": [
        'Fill in the blank: "Ich sehe ___ Bruder." You can answer with just the missing word.',
        'Complete this sentence in German: "Sie kauft ___ Apfel." Write only the missing article if you want.',
        'Write one short German sentence with the phrase "einen Freund".',
        'Fill in the blank with the correct article: "Wir besuchen ___ Lehrer."',
    ],
    "negation_kein": [
        'Fill in the blank with the correct negation: "Das ist ___ Apfel." You can answer with just the missing word.',
        'Complete this German sentence with kein/keine/keinen: "Ich habe ___ Auto."',
        'Write one short German sentence saying that you do not have a dog. Use "kein".',
        'Fill in the blank: "Wir haben heute ___ Zeit."',
    ],
    "present_tense_basic_verbs": [
        'Write one complete German sentence about something you do every day. Start with "Ich".',
        'Write one German sentence about where you live or what you study.',
        'Tell me one thing you do in the morning in German. Use present tense.',
        'Write one simple German sentence with "ich" and a correctly conjugated verb.',
    ],
    "perfect_tense_basics": [
        'Write one German sentence about what you did yesterday. Use Perfekt, for example with "habe" or "bin".',
        'Tell me in German one thing you did last weekend. Use Perfekt.',
        'Write one German sentence with "Ich habe..." and a past participle.',
        'Write one German sentence about a place you went to recently. Use Perfekt with "bin".',
    ],
    "accusative_with_movement": [
        'Write one German sentence showing movement toward a place, using a phrase like "auf den", "in die", or "an das".',
        'Write a sentence where you put something onto a table. Use German.',
        'Write one German sentence with movement and "in die" or "auf den".',
        'Complete a sentence about placing a book somewhere, using a two-way preposition with movement.',
    ],
    "comparatives_basics": [
        'Write one German sentence comparing two things. Use a comparative form and "als".',
        'Compare a car and a bicycle in German using "...er als".',
        'Write one sentence in German saying that one city is bigger, smaller, or nicer than another.',
        'Use "als" in one German comparison sentence.',
    ],
    "subordinate_clause_weil": [
        'Answer in German: Why are you learning German? Use "weil" and put the verb at the end of the weil-clause.',
        'Write one German sentence explaining why you study today. Use "weil".',
        'Complete this idea in German: "Ich lerne Deutsch, weil..." Put the verb at the end.',
        'Write one reason sentence in German with "weil".',
    ],
    "konjunktiv_ii_basics": [
        'Write one German sentence about what you would do if you had more free time. Use "würde", "hätte", or "wäre".',
        'Write one polite German request with "könnte" or "würde".',
        'Say in German what you would buy if you had more money. Use Konjunktiv II.',
        'Write one hypothetical German sentence with "würde".',
    ],
    "relative_clauses_basics": [
        'Combine this idea into one German sentence with a relative clause: "Das ist die Person. Die Person hilft mir."',
        'Write one German sentence describing a person with "der", "die", or "das" as a relative pronoun.',
        'Complete this sentence with a relative clause: "Das ist der Freund, ..."',
        'Write one German sentence about a thing or person you like, using a relative clause.',
    ],
}


def fallback_dynamic_diagnostic_question(task):
    variants = DIAGNOSTIC_QUESTION_VARIANTS.get(task.get("grammar_point", ""))
    if not variants:
        return task.get("question") or "Please write one short answer in German."
    return random.choice(variants)


def is_bad_diagnostic_question(question, task):
    if not question or len(question.strip()) < 12:
        return True
    lowered = question.lower()
    blocked_phrases = [
        "check whether",
        "diagnostic goal",
        "criteria:",
        "reference answer",
        "grammar point",
        "target learner band",
        task.get("example_answer", "").lower(),
    ]
    return any(phrase and phrase in lowered for phrase in blocked_phrases)


def generate_diagnostic_question(task, user_level="A1"):
    fallback_question = fallback_dynamic_diagnostic_question(task)

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
- Do not include the reference answer or an answer key.
- Keep it short.
- Do not label difficulty or mention CEFR.
- Make the expected answer format clear: say whether the learner should write a full sentence or only fill the blank.
- Avoid trick questions; test only the listed grammar point.
- Use a fresh scenario, noun, or context instead of repeating the same wording every time.
- Never expose internal phrases like "check whether", "diagnostic goal", "criteria", or "grammar point".

Reply with only the question text.
""".strip()

    generated_question = invoke_llm_content(
        prompt,
        fallback_text=fallback_question,
        retries=0,
        call_name=f"diagnostic:question:{task.get('grammar_point', 'unknown')}",
    )
    if is_bad_diagnostic_question(generated_question, task):
        return fallback_question
    return generated_question


def deterministic_diagnostic_feedback(task, evaluation):
    topic = task.get("topic", "German grammar")
    grammar_point = task.get("grammar_point", "")

    if evaluation["score_value"] == 2:
        if grammar_point == "indefinite_articles_ein_eine_einen":
            return "Great, that accusative article is exactly what we needed here."
        if grammar_point == "negation_kein":
            return "Nice work, you used the negation with kein correctly."
        if grammar_point == "present_tense_basic_verbs":
            return "Good sentence, the present-tense verb is clear."
        if grammar_point == "perfect_tense_basics":
            return "Great, that shows a clear Perfekt structure."
        if grammar_point == "accusative_with_movement":
            return "Nice, you showed movement with the preposition clearly."
        if grammar_point == "comparatives_basics":
            return "Good comparison, the comparative pattern with als is clear."
        if grammar_point == "subordinate_clause_weil":
            return "Great, your weil-clause shows the sentence structure we are checking."
        if grammar_point == "konjunktiv_ii_basics":
            return "Nice, that hypothetical form works well here."
        if grammar_point == "relative_clauses_basics":
            return "Good, that relative clause connects the ideas clearly."
        return f"Nice work, your {topic} answer is strong."

    if evaluation["score_value"] == 1:
        return f"Good start, I can see part of the {topic} pattern there. We'll strengthen it step by step."

    return f"Good try. This {topic} point still needs practice, and that is completely okay."


def build_human_diagnostic_feedback(task, evaluation, user_level="A1"):
    return deterministic_diagnostic_feedback(task, evaluation)


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
    topic_hint = state.get("topic_hint", "Sentence structure")
    latest_user_message = state.get("latest_user_message", "")
    user_level = state.get("user_level", "A1")
    syllabus_reference = select_syllabus_reference(
        level=user_level,
        topic=topic_hint,
        grammar_point=state.get("grammar_point", ""),
        learner_profile=state.get("learner_profile", default_learner_profile()),
    )

    if goal_type == "study_plan":
        return {
            "retrieved_context": "",
            "retrieved_documents": [],
            "retrieval_used_fallback": False,
            "retrieval_focus": {},
            "syllabus_reference": syllabus_reference,
            "topic_hint": topic_hint,
            "grammar_point": "",
        }

    retrieval_query = latest_user_message
    if syllabus_reference.get("search_query"):
        retrieval_query = f"{latest_user_message}\nCurriculum reference: {syllabus_reference['search_query']}"

    bundle = retrieve_context_bundle(
        query=retrieval_query,
        user_level=user_level,
        topic_hint=topic_hint,
        k=5,
        learner_profile=state.get("learner_profile", default_learner_profile()),
        grammar_point_mastery=state.get("grammar_point_mastery", default_grammar_point_mastery()),
    )
    return {
        "topic_hint": bundle["topic"],
        "grammar_point": bundle.get("grammar_point", ""),
        "retrieved_context": bundle["context_text"],
        "retrieved_documents": bundle["documents"],
        "retrieval_used_fallback": bundle["used_fallback"],
        "retrieval_focus": bundle.get("retrieval_focus", {}),
        "syllabus_reference": syllabus_reference,
    }


def plan_lesson(state):
    level = state.get("user_level", "A1")
    goal_type = state.get("goal_type", "general_help")
    response_style = state.get("response_style", "gentle")
    language_support = state.get("language_support", "mostly_english")
    practice_now = state.get("practice_now", "NO")
    syllabus_reference = state.get("syllabus_reference") or select_syllabus_reference(
        level=level,
        topic=state.get("topic_hint", "Sentence structure"),
        grammar_point=state.get("grammar_point", ""),
        learner_profile=state.get("learner_profile", default_learner_profile()),
    )

    return {
        "lesson_plan": {
            "level_guideline": LEVEL_GUIDELINES.get(level, LEVEL_GUIDELINES["A1"]),
            "goal_type": goal_type,
            "response_style": response_style,
            "language_support": language_support,
            "practice_now": practice_now,
            "topic": state.get("topic_hint", "Sentence structure"),
            "grammar_point": state.get("grammar_point", ""),
            "use_retrieval_fallback": state.get("retrieval_used_fallback", False),
            "syllabus_reference": syllabus_reference,
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

    syllabus_reference = state.get("lesson_plan", {}).get("syllabus_reference", {})
    lesson_id = syllabus_reference.get("lesson_id")
    if lesson_id:
        syllabus_history = list(profile.get("syllabus_history", []))
        syllabus_history.append(lesson_id)
        profile["syllabus_history"] = unique_keep_order(syllabus_history)[-10:]
        profile["current_syllabus_lesson"] = syllabus_reference

    return profile


def summarize_learner_profile(profile):
    if not profile:
        return "No learner profile is available."

    current_goal = profile.get("current_goal") or "not clearly set yet"
    recent_topics = ", ".join(profile.get("recent_topics", [])) or "none yet"
    recent_grammar_points = ", ".join(profile.get("recent_grammar_points", [])) or "none yet"
    weak_topics = ", ".join(profile.get("weak_topics", [])) or "none identified yet"
    strong_topics = ", ".join(profile.get("strong_topics", [])) or "none identified yet"
    current_syllabus = profile.get("current_syllabus_lesson", {})
    current_syllabus_text = (
        f"{current_syllabus.get('level', '')} {current_syllabus.get('lesson_id', '')} "
        f"{current_syllabus.get('title', '')}"
    ).strip() or "none yet"
    preferred_language_support = profile.get("preferred_language_support", "mostly_english")
    last_goal_type = profile.get("last_goal_type") or "unknown"

    return (
        f"Current goal: {current_goal}. "
        f"Recent topics: {recent_topics}. "
        f"Recent grammar points: {recent_grammar_points}. "
        f"Weak topics: {weak_topics}. "
        f"Strong topics: {strong_topics}. "
        f"Current syllabus reference: {current_syllabus_text}. "
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


def get_next_cefr_level(level):
    try:
        current_index = CEFR_LEVEL_ORDER.index(level)
    except ValueError:
        return None

    next_index = current_index + 1
    if next_index >= len(CEFR_LEVEL_ORDER):
        return None
    return CEFR_LEVEL_ORDER[next_index]


def level_mastery_status(level, grammar_point_mastery):
    required_points = CEFR_PROMOTION_REQUIREMENTS.get(level, [])
    mastery = grammar_point_mastery or {}
    mastered_points = [
        grammar_point
        for grammar_point in required_points
        if mastery.get(grammar_point, 0) >= CEFR_PROMOTION_MASTERY_THRESHOLD
    ]
    missing_points = [
        grammar_point
        for grammar_point in required_points
        if grammar_point not in mastered_points
    ]
    return {
        "level": level,
        "required_points": required_points,
        "mastered_points": mastered_points,
        "missing_points": missing_points,
        "ready_for_promotion": bool(required_points) and not missing_points,
    }


def evaluate_cefr_progression(state, grammar_point_mastery):
    current_level = state.get("user_level", "A1")
    next_level = get_next_cefr_level(current_level)

    status = level_mastery_status(current_level, grammar_point_mastery)
    progression = {
        "level_progression_status": status,
        "level_promoted": False,
    }

    if not next_level or not status["ready_for_promotion"]:
        return progression

    return {
        **progression,
        "level_promoted": True,
        "previous_level": current_level,
        "new_level": next_level,
        "promotion_reason": (
            f"All required {current_level} grammar points reached mastery "
            f"{CEFR_PROMOTION_MASTERY_THRESHOLD}+."
        ),
    }


def apply_level_progression_to_profile(profile, progression):
    if not progression.get("level_promoted"):
        return profile

    profile = dict(profile or default_learner_profile())
    history = list(profile.get("level_progression_history", []))
    history.append(
        {
            "from_level": progression["previous_level"],
            "to_level": progression["new_level"],
            "reason": progression["promotion_reason"],
        }
    )
    profile["level_progression_history"] = history[-10:]
    return profile


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
    syllabus_reference_text = format_syllabus_reference(lesson_plan.get("syllabus_reference", {}))
    level_source = state.get("level_source", "diagnostic")
    level_confidence = state.get("level_confidence", "high")

    style_instruction_map = {
        "gentle": "Use a warm, encouraging, human tone.",
        "structured": "Be very clear and well-organized.",
        "brief": "Keep the answer concise but still supportive.",
    }

    return f"""
You are Lexi, a warm adaptive German tutor.

Learner: level {level} ({level_source}, confidence {level_confidence}).
Topic: {lesson_plan.get('topic', 'Sentence structure')}.
Grammar point: {grammar_point or 'not clearly identified'}.
Style: {style_instruction_map.get(response_style, style_instruction_map['gentle'])}
Language: {build_language_support_instructions(language_support)}

Learner memory:
{learner_profile_summary}
{mastery_summary}

Diagnostic summary:
{diagnostic_summary}

Syllabus guide:
{syllabus_reference_text}

Rules:
- Answer the learner's request first.
- Use the syllabus only as a gentle reference, not a restriction.
- Keep the response concise, human, encouraging, and level-appropriate.
- Use short German examples with English support unless the learner asks for more German.
- Teach one small focus at a time and end with one comfortable next step.
- If the learner says they understood, says "continue", or asks you to teach, continue the current lesson thread with the next small exercise.
- Do not repeatedly ask the learner what topic they want; as the guided tutor, choose a sensible next step from memory, syllabus, or recent mistakes.
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

    response = invoke_llm_content(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        fallback_text=build_service_fallback_response(state),
        retries=0,
        call_name=f"answer:{state.get('goal_type', 'branch')}",
    )
    return {"draft_response": response}


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

        response = invoke_llm_content(
            prompt,
            fallback_text=(
                "That makes sense. We can adjust the difficulty gently, or you can retake "
                "the quick level check if you want a cleaner reset."
            ),
            retries=1,
            call_name="answer:level_adjustment_unclear",
        )
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

    response = invoke_llm_content(
        prompt,
        fallback_text=(
            f"Thanks, {display_name}. I will adapt our lessons to {requested_level} from now on, "
            "and we can slow down or speed up whenever it feels right."
        ),
        retries=1,
        call_name="answer:level_adjustment_confirm",
    )
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
    return {
        "quality_status": "PASS",
        "quality_rationale": "Deterministic quality gate passed; no extra model call needed.",
    }


def answer_revision(state):
    draft_response = state.get("draft_response", "")
    quality_rationale = state.get("quality_rationale", "Please improve the response.")
    level = state.get("user_level", "A1")
    goal_type = state.get("goal_type", "general_help")
    language_support = state.get("language_support", "mostly_english")
    syllabus_reference_text = format_syllabus_reference(
        state.get("lesson_plan", {}).get("syllabus_reference", {})
    )

    prompt = f"""
You are revising a tutor response for a German learner.

Student level: {level}
Goal type: {goal_type}
Language support: {language_support}
Syllabus reference:
{syllabus_reference_text}

Original draft:
{draft_response}

Reviewer feedback:
{quality_rationale}

Revise the response so it:
- fits the learner's level
- sounds warm and human
- uses enough English support for the learner
- matches the learner's actual request
- uses the syllabus as a gentle curriculum reference when useful
- does not start practice unexpectedly

Reply with only the improved tutor response.
""".strip()

    response = invoke_llm_content(
        prompt,
        fallback_text=draft_response,
        retries=0,
        call_name="review:answer_revision",
    )
    return {"draft_response": response}


def finalize_response(state):
    draft_response = state.get("draft_response", "")
    if not draft_response.strip():
        draft_response = build_service_fallback_response(state)
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
    progression = evaluate_cefr_progression(state, grammar_point_mastery)
    profile = apply_level_progression_to_profile(profile, progression)

    updated = {
        "learner_profile": profile,
        "grammar_point_mastery": grammar_point_mastery,
        "level_progression_status": progression.get("level_progression_status", {}),
        "level_promoted": progression.get("level_promoted", False),
    }
    if progression.get("level_promoted"):
        updated.update(
            {
                "user_level": progression["new_level"],
                "level_source": "automatic_mastery_progression",
                "level_confidence": "medium",
                "level_change_rationale": progression["promotion_reason"],
            }
        )

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
        "level_progression_status": {},
        "level_promoted": False,
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
        "level_progression_status": saved.get("level_progression_status", {}),
        "level_promoted": False,
    }


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_LEXI = "\033[96m"
ANSI_YOU = "\033[92m"


def speaker_label(label: str, color: str) -> str:
    return f"{ANSI_BOLD}{color}{label}:{ANSI_RESET}"


LEXI_LABEL = speaker_label("LEXI", ANSI_LEXI)
YOU_LABEL = speaker_label("YOU", ANSI_YOU)


if __name__ == "__main__":
    print("\n")
    learner_name = input("Learner name: ").strip()
    while not learner_name:
        learner_name = input("Learner name: ").strip()

    learner_id = learner_name.lower()

    if learner_exists(learner_id):
        print("\n")
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
            print(f"\n{LEXI_LABEL} {current_state['messages'][-1]['content']}")
        else:
            current_state = build_initial_state(learner_id, learner_name)
            current_state["is_returning_learner"] = True
            current_state["wants_retake_diagnostic"] = True
            current_state = app.invoke(current_state)
            print(f"\n{LEXI_LABEL} {current_state['messages'][-1]['content']}")
    else:
        current_state = build_initial_state(learner_id, learner_name)
        current_state = app.invoke(current_state)
        print(f"\n{LEXI_LABEL} {current_state['messages'][-1]['content']}")

    while True:
        user_text = input(f"{YOU_LABEL} ").strip()
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
        print(f"\n{LEXI_LABEL} {current_state['messages'][-1]['content']}")
