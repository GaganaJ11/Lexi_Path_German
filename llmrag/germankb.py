import re
import json
import os
import hashlib
from typing import List, Dict, Optional

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter



OUTPUT_FILE = "LexiPath_Clean_Knowledge_Base.jsonl"
MAX_ROWS_PER_DATASET = 1500

CHUNK_SIZE = 450
CHUNK_OVERLAP = 60

MIN_TEXT_LEN = 80
MAX_TEXT_LEN = 2000

NICO_FILE_PATH = r"/root/LexiPath_Data/raw_data/DW-Nicos-Weg/Nico's Weg A1,A2,B1 - FULL SCRIPT.txt"



def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)

    # normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove repeated separators
    text = re.sub(r"[=\-*_]{4,}", " ", text)

    # remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # remove too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # strip
    text = text.strip()

    return text


def looks_like_noise(text: str) -> bool:
    if not text:
        return True

    t = text.strip()

    # too short
    if len(t) < MIN_TEXT_LEN:
        return True

    # too symbolic / bad extraction
    symbol_ratio = sum(1 for c in t if not c.isalnum() and not c.isspace()) / max(len(t), 1)
    if symbol_ratio > 0.25:
        return True

    # too few letters
    alpha_ratio = sum(1 for c in t if c.isalpha()) / max(len(t), 1)
    if alpha_ratio < 0.5:
        return True

    return False



GERMAN_STOPWORDS_HINT = [
    "ich", "du", "er", "sie", "wir", "ihr",
    "ist", "bin", "sind", "habe", "hast",
    "der", "die", "das", "ein", "eine",
    "und", "oder", "weil", "dass", "nicht"
]

GRAMMAR_KEYWORDS = {
    "articles": ["artikel", "der", "die", "das", "ein", "eine", "den", "dem"],
    "verbs": ["verb", "konjugation", "bin", "bist", "ist", "habe", "hast", "sein", "haben"],
    "cases": ["akkusativ", "dativ", "nominativ", "genitiv"],
    "pronouns": ["pronomen", "ich", "du", "er", "sie", "wir", "ihr"],
    "connectors": ["weil", "dass", "wenn", "obwohl", "deshalb"],
    "sentence_structure": ["satz", "satzbau", "wortstellung", "frage", "w-frage"],
    "prepositions": ["präposition", "mit", "für", "von", "zu", "bei", "nach"]
}

COMMUNICATION_KEYWORDS = {
    "restaurant": ["restaurant", "bestellen", "speisekarte", "rechnung"],
    "introductions": ["heiße", "komme aus", "wohne", "vorstellen"],
    "daily_life": ["morgen", "tag", "arbeit", "schule", "familie"],
    "directions": ["wo ist", "wie komme ich", "straße", "bahnhof"],
    "shopping": ["kaufen", "preis", "kosten", "geschäft"]
}


def quality_score(text: str) -> float:
    score = 0.0
    t = text.lower()

    # length
    if 120 <= len(t) <= 800:
        score += 0.25
    elif 80 <= len(t) <= 1200:
        score += 0.18

    # looks German-ish
    stopword_hits = sum(1 for w in GERMAN_STOPWORDS_HINT if w in t)
    if stopword_hits >= 4:
        score += 0.25
    elif stopword_hits >= 2:
        score += 0.15

    # full sentences
    sentence_count = len(re.findall(r"[.!?]", t))
    if sentence_count >= 2:
        score += 0.20
    elif sentence_count >= 1:
        score += 0.10

    # grammar / learning relevance
    grammar_hits = sum(
        any(k in t for k in kw_list)
        for kw_list in GRAMMAR_KEYWORDS.values()
    )
    comm_hits = sum(
        any(k in t for k in kw_list)
        for kw_list in COMMUNICATION_KEYWORDS.values()
    )

    if grammar_hits >= 1 or comm_hits >= 1:
        score += 0.20

    # penalize obvious noise
    if looks_like_noise(text):
        score -= 0.40

    return max(0.0, min(1.0, score))


def is_good_learning_text(text: str) -> bool:
    if looks_like_noise(text):
        return False
    if quality_score(text) < 0.35:
        return False
    return True


def infer_level(text: str, fallback: str = "A1") -> str:
    t = text.lower()

    b1_markers = [
        "obwohl", "trotzdem", "einerseits", "andererseits",
        "meiner meinung nach", "ich denke, dass", "vorteile", "nachteile"
    ]

    a2_markers = [
        "weil", "dass", "wenn", "deshalb",
        "gegangen", "gemacht", "gesehen", "gesprochen",
        "sich ", "mich ", "dich "
    ]

    a1_markers = [
        "ich heiße", "ich bin", "du bist", "er ist",
        "ich wohne", "ich komme aus", "das ist",
        "der", "die", "das", "ein", "eine"
    ]

    if any(m in t for m in b1_markers):
        return "B1"
    if any(m in t for m in a2_markers):
        return "A2"
    if any(m in t for m in a1_markers):
        return "A1"

    return fallback


# =========================================================
# TOPIC / SKILL INFERENCE
# =========================================================
def infer_topic(text: str) -> str:
    t = text.lower()

    for topic, kws in GRAMMAR_KEYWORDS.items():
        if any(k in t for k in kws):
            return topic

    for topic, kws in COMMUNICATION_KEYWORDS.items():
        if any(k in t for k in kws):
            return topic

    return "general"


def infer_skill(text: str) -> str:
    t = text.lower()

    grammar_hits = sum(any(k in t for k in kws) for kws in GRAMMAR_KEYWORDS.values())
    comm_hits = sum(any(k in t for k in kws) for kws in COMMUNICATION_KEYWORDS.values())

    if grammar_hits > 0 and grammar_hits >= comm_hits:
        return "grammar"
    if comm_hits > 0:
        return "communication"
    return "general"


# =========================================================
# CHUNKING
# =========================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)


def chunk_text(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return text_splitter.split_text(text)


# =========================================================
# DEDUP
# =========================================================
def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


# =========================================================
# RECORD BUILDING
# =========================================================
def make_record(
    text: str,
    source: str,
    level: Optional[str] = None,
    skill: Optional[str] = None,
    topic: Optional[str] = None,
    doc_type: str = "lesson_chunk"
) -> Optional[Dict]:
    text = normalize_text(text)

    if not is_good_learning_text(text):
        return None

    q = quality_score(text)

    return {
        "text": text,
        "metadata": {
            "source": source,
            "level": level or infer_level(text),
            "skill": skill or infer_skill(text),
            "topic": topic or infer_topic(text),
            "doc_type": doc_type,
            "quality_score": round(q, 3)
        }
    }


# =========================================================
# DATASET EXTRACTION
# =========================================================
def extract_text_from_row(row: Dict) -> List[str]:
    candidates = []

    for key in ["context", "text", "question", "answer", "instruction", "response"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    # combine useful pairs
    if row.get("question") and row.get("answer"):
        qa = f"Question: {row['question']}\nAnswer: {row['answer']}"
        candidates.append(qa)

    return candidates


def process_hf_dataset(dataset_obj, source_label: str) -> List[Dict]:
    output = []
    seen = set()
    count = 0

    rows = dataset_obj["train"]

    for row in rows:
        texts = extract_text_from_row(row)

        for raw_text in texts:
            raw_text = normalize_text(raw_text)

            if not is_good_learning_text(raw_text):
                continue

            chunks = chunk_text(raw_text)

            for chunk in chunks:
                record = make_record(
                    text=chunk,
                    source=source_label,
                    doc_type="dataset_chunk"
                )
                if record is None:
                    continue

                h = text_hash(record["text"])
                if h in seen:
                    continue

                seen.add(h)
                output.append(record)

        count += 1
        if count >= MAX_ROWS_PER_DATASET:
            break

    return output


# =========================================================
# NICOS WEG EXTRACTION
# =========================================================
def extract_nicos_lessons(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        full_content = f.read()

    raw_lessons = re.split(r"={10,}", full_content)
    lessons = []

    for lesson in raw_lessons:
        lesson = lesson.strip()
        if not lesson:
            continue

        lines = lesson.split("\n")
        header = lines[0].strip()

        match = re.match(r"(A1|A2|B1)-(\d+)-(.*)", header)
        if not match:
            continue

        level, number, title = match.groups()
        body = "\n".join(lines[1:]).strip()
        body = normalize_text(body)

        if len(body) < MIN_TEXT_LEN:
            continue

        lessons.append({
            "lesson_id": f"{level}-{number}",
            "level": level,
            "title": title.strip(),
            "text": body
        })

    return lessons


def process_nicos_lessons(file_path: str) -> List[Dict]:
    output = []
    seen = set()

    lessons = extract_nicos_lessons(file_path)

    for lesson in lessons:
        chunks = chunk_text(lesson["text"])

        for chunk in chunks:
            record = make_record(
                text=chunk,
                source="Nicos-Weg",
                level=lesson["level"],
                topic=infer_topic(chunk),
                skill=infer_skill(chunk),
                doc_type="lesson_chunk"
            )

            if record is None:
                continue

            record["metadata"]["lesson_id"] = lesson["lesson_id"]
            record["metadata"]["lesson_title"] = lesson["title"]

            h = text_hash(record["text"])
            if h in seen:
                continue

            seen.add(h)
            output.append(record)

    return output


# =========================================================
# GLOBAL DEDUP + FILTER
# =========================================================
def deduplicate_records(records: List[Dict]) -> List[Dict]:
    seen = set()
    cleaned = []

    for rec in records:
        h = text_hash(rec["text"])
        if h in seen:
            continue
        seen.add(h)
        cleaned.append(rec)

    return cleaned


def final_filter(records: List[Dict]) -> List[Dict]:
    cleaned = []
    for rec in records:
        q = rec["metadata"].get("quality_score", 0.0)
        txt = rec["text"]

        if q < 0.40:
            continue
        if len(txt) < MIN_TEXT_LEN or len(txt) > MAX_TEXT_LEN:
            continue
        cleaned.append(rec)

    return cleaned


# =========================================================
# EXPORT
# =========================================================
def export_jsonl(records: List[Dict], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


# =========================================================
# MAIN
# =========================================================
def main():
    all_records = []

    print("Loading datasets...")

    ds_avemio_qa = load_dataset("avemio/German-RAG-CPT-HESSIAN-AI", "question-answering")
    ds_avemio_de = load_dataset("avemio/German-RAG-CPT-HESSIAN-AI", "reasoning-de")
    ds_avemio_en = load_dataset("avemio/German-RAG-CPT-HESSIAN-AI", "reasoning-en")
    ds_disco = load_dataset("DiscoResearch/germanrag")

    print("Processing Avemio QA...")
    all_records.extend(process_hf_dataset(ds_avemio_qa, "Avemio-QA"))

    print("Processing Avemio Reasoning DE...")
    all_records.extend(process_hf_dataset(ds_avemio_de, "Avemio-Reasoning-DE"))

    print("Processing Avemio Reasoning EN...")
    all_records.extend(process_hf_dataset(ds_avemio_en, "Avemio-Reasoning-EN"))

    print("Processing DiscoResearch...")
    all_records.extend(process_hf_dataset(ds_disco, "DiscoResearch"))

    print("Processing Nicos Weg...")
    all_records.extend(process_nicos_lessons(NICO_FILE_PATH))

    print(f"Before dedup: {len(all_records)}")
    all_records = deduplicate_records(all_records)
    print(f"After dedup: {len(all_records)}")

    all_records = final_filter(all_records)
    print(f"After final filter: {len(all_records)}")

    export_jsonl(all_records, OUTPUT_FILE)
    print(f"Saved clean knowledge base to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()