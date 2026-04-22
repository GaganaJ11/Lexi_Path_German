import json
import re
from typing import Dict, Iterable, List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

CONNECTION_STRING = "postgresql+psycopg://postgres:mypassword@localhost:5432/postgres"
COLLECTION_NAME = "lexipath_grammar_v2"
EMBEDDING_MODEL = "embeddinggemma"
SOURCE_FILE = "LexiPath_Final_Knowledge_Base.jsonl"

RULE_SOURCES = {"DiscoResearch"}
EXAMPLE_SOURCES = {"Nicos-Weg-GitHub"}
OPTIONAL_RULE_SOURCES = {"Avemio_ReasoningDE"}
EXCLUDED_SOURCES = {"Avemio_QuestionAnswering", "Avemio_ReasoningEN"}

TOPIC_TO_GRAMMAR_POINT = {
    "Articles": "definite_articles_basics",
    "Negation": "negation_kein",
    "Verb Conjugation": "present_tense_basic_verbs",
    "Sentence Structure": "subordinate_clause_weil",
    "Cases": "accusative_with_movement",
    "Grammar": "general_grammar",
}

MANUAL_RULES = [
    {
        "level": "A1",
        "topic": "Articles",
        "grammar_point": "accusative_masculine_den",
        "content": (
            "Use 'den' for masculine nouns in the accusative case. "
            "The nominative form 'der' changes to 'den' when the noun is the direct object. "
            "Example: Der Mann ist hier. Ich sehe den Mann."
        ),
    },
    {
        "level": "A1",
        "topic": "Articles",
        "grammar_point": "definite_articles_basics",
        "content": (
            "German definite articles are 'der', 'die', and 'das'. "
            "You should learn each noun together with its article."
        ),
    },
    {
        "level": "A1",
        "topic": "Articles",
        "grammar_point": "indefinite_articles_ein_eine_einen",
        "content": (
            "Use 'ein' for many masculine or neuter nouns, 'eine' for many feminine nouns, "
            "and 'einen' for masculine nouns in the accusative."
        ),
    },
    {
        "level": "A1",
        "topic": "Negation",
        "grammar_point": "negation_kein",
        "content": (
            "Use 'kein' to negate nouns with an indefinite meaning. "
            "Example: Ich habe ein Auto. -> Ich habe kein Auto."
        ),
    },
    {
        "level": "A1",
        "topic": "Verb Conjugation",
        "grammar_point": "present_tense_basic_verbs",
        "content": (
            "In a simple German sentence, the conjugated verb usually comes in position two. "
            "Example: Ich wohne in Berlin."
        ),
    },
    {
        "level": "A2",
        "topic": "Verb Conjugation",
        "grammar_point": "perfect_tense_basics",
        "content": (
            "German Perfekt is usually formed with 'haben' or 'sein' plus a past participle. "
            "Example: Ich habe Deutsch gelernt."
        ),
    },
    {
        "level": "A2",
        "topic": "Cases",
        "grammar_point": "accusative_with_movement",
        "content": (
            "When a two-way preposition shows movement toward a destination, German often uses the accusative case. "
            "Example: Ich lege das Buch auf den Tisch."
        ),
    },
    {
        "level": "A2",
        "topic": "Grammar",
        "grammar_point": "comparatives_basics",
        "content": (
            "To compare two things in German, use the comparative form plus 'als'. "
            "Example: Ein Auto ist schneller als ein Fahrrad."
        ),
    },
    {
        "level": "B1",
        "topic": "Sentence Structure",
        "grammar_point": "subordinate_clause_weil",
        "content": (
            "In a subordinate clause with 'weil', the conjugated verb goes to the end. "
            "Example: Ich lerne Deutsch, weil ich in Deutschland arbeiten will."
        ),
    },
    {
        "level": "B1",
        "topic": "Grammar",
        "grammar_point": "konjunktiv_ii_basics",
        "content": (
            "Konjunktiv II is often used to talk about hypothetical situations, wishes, or polite ideas. "
            "A common pattern is 'würde' plus infinitive. Example: Ich würde viel reisen."
        ),
    },
    {
        "level": "B1",
        "topic": "Sentence Structure",
        "grammar_point": "relative_clauses_basics",
        "content": (
            "A relative clause adds information about a noun and often begins with a relative pronoun like 'die' or 'der'. "
            "Example: Das ist die Frau, die mir hilft."
        ),
    },
]


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def looks_like_rule_text(text: str) -> bool:
    lowered = text.lower()

    strong_grammar_markers = [
        "akkusativ", "dativ", "nominativ", "genitiv",
        "artikel", "article", "verb", "konjugation", "perfekt", "partizip",
        "nebensatz", "wortstellung", "grammatik", "regel", "beispiel",
        "maskulin", "feminin", "neutral", "objekt", "subjekt",
        "komparativ", "vergleich", "konjunktiv", "relativsatz", "kein", "verneinung",
    ]

    non_rule_markers = [
        "welche", "wer", "was", "wann", "wo", "warum", "wie viele",
        "question", "kontext:", "context:",
        "north carolina", "hessian", "rail", "bahnstrecken",
    ]

    if any(marker in lowered for marker in non_rule_markers):
        return False

    return any(marker in lowered for marker in strong_grammar_markers)


def source_allows_rule(source: str, text: str) -> bool:
    lowered = text.lower()

    if source == "DiscoResearch":
        required = [
            "akkusativ", "dativ", "nominativ", "genitiv", "artikel",
            "verb", "konjugation", "perfekt", "nebensatz", "wortstellung",
            "grammatik", "komparativ", "konjunktiv", "relativsatz", "kein",
        ]
        return any(marker in lowered for marker in required)

    if source == "Avemio_ReasoningDE":
        required = [
            "akkusativ", "dativ", "nominativ", "genitiv", "artikel",
            "grammatik", "regel", "verb", "perfekt", "komparativ",
            "konjunktiv", "relativsatz", "verneinung", "kein",
        ]
        return any(marker in lowered for marker in required)

    return True


def infer_grammar_point(topic: str, text: str) -> str:
    lowered = normalize_text(text)
    tokens = re.findall(r"[a-zA-ZäöüÄÖÜß]+", lowered)
    padded = f" {lowered} "

    if topic == "Articles":
        if "den" in tokens and "denn" not in tokens:
            if any(word in lowered for word in ["akkusativ", "objekt", "maskulin", "artikel"]) or len(tokens) <= 8:
                return "accusative_masculine_den"
        if any(token in tokens for token in ("ein", "eine", "einen")):
            return "indefinite_articles_ein_eine_einen"
        return "definite_articles_basics"

    if topic == "Negation":
        return "negation_kein"

    if topic == "Verb Conjugation":
        if any(token in tokens for token in ("perfekt", "partizip", "haben", "sein")):
            return "perfect_tense_basics"
        return "present_tense_basic_verbs"

    if topic == "Cases":
        return "accusative_with_movement"

    if topic == "Sentence Structure":
        if "relativsatz" in tokens or "relative" in tokens:
            return "relative_clauses_basics"
        return "subordinate_clause_weil"

    if topic == "Grammar":
        if "kein" in tokens or "keine" in tokens:
            return "negation_kein"
        if "komparativ" in tokens or "als" in tokens:
            return "comparatives_basics"
        if "konjunktiv" in tokens or "würde" in tokens or "wuerde" in tokens:
            return "konjunktiv_ii_basics"
        if "relativsatz" in tokens or "relative" in tokens:
            return "relative_clauses_basics"
        return "general_grammar"

    return TOPIC_TO_GRAMMAR_POINT.get(topic, "general_grammar")


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def split_dialogue_units(text: str) -> List[str]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    if len(blocks) > 1:
        return blocks

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    units = []
    current = []

    for line in lines:
        if re.match(r"^[A-ZÄÖÜ][A-ZÄÖÜa-zäöüß .'-]{1,30}:$", line):
            if current:
                units.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        units.append(" ".join(current).strip())

    return units or [text.strip()]


def build_manual_rule_documents() -> List[Document]:
    docs = []
    for rule in MANUAL_RULES:
        docs.append(
            Document(
                page_content=rule["content"],
                metadata={
                    "level": rule["level"],
                    "source": "LexiPath_ManualRules",
                    "topic": rule["topic"],
                    "lesson_id": "manual",
                    "chunk_type": "rule",
                    "grammar_point": rule["grammar_point"],
                },
            )
        )
    return docs


def build_rule_chunks(text: str, metadata: Dict[str, str]) -> List[Document]:
    chunks = []
    seen = set()

    for sentence in split_sentences(text):
        cleaned = clean_text(sentence)
        if len(cleaned) < 40 or len(cleaned) > 500:
            continue
        if not looks_like_rule_text(cleaned):
            continue
        if not source_allows_rule(metadata["source"], cleaned):
            continue

        normalized = normalize_text(cleaned)
        if normalized in seen:
            continue
        seen.add(normalized)

        grammar_point = infer_grammar_point(metadata["topic"], cleaned)
        chunk_metadata = {
            **metadata,
            "chunk_type": "rule",
            "grammar_point": grammar_point,
        }
        chunks.append(Document(page_content=cleaned, metadata=chunk_metadata))

    return chunks


def build_example_chunks(text: str, metadata: Dict[str, str]) -> List[Document]:
    chunks = []
    seen = set()

    for unit in split_dialogue_units(text):
        for sentence in split_sentences(unit):
            cleaned = clean_text(sentence)
            if len(cleaned) < 15 or len(cleaned) > 260:
                continue

            normalized = normalize_text(cleaned)
            if normalized.count(":") > 1:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)

            grammar_point = infer_grammar_point(metadata["topic"], cleaned)
            chunk_metadata = {
                **metadata,
                "chunk_type": "example",
                "grammar_point": grammar_point,
            }
            chunks.append(Document(page_content=cleaned, metadata=chunk_metadata))

    return chunks


def load_source_records(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_documents(path: str) -> List[Document]:
    all_documents: List[Document] = build_manual_rule_documents()

    for record in load_source_records(path):
        metadata = record.get("metadata", {})
        source = str(metadata.get("source", "unknown"))
        topic = str(metadata.get("topic", "Grammar"))
        level = str(metadata.get("level", "A1"))
        lesson_id = str(metadata.get("lesson_id", "unknown"))

        if source in EXCLUDED_SOURCES:
            continue

        cleaned_content = clean_text(record.get("content", ""))
        if len(cleaned_content) < 20:
            continue

        base_metadata = {
            "level": level,
            "source": source,
            "topic": topic,
            "lesson_id": lesson_id,
        }

        if source in RULE_SOURCES:
            all_documents.extend(build_rule_chunks(cleaned_content, base_metadata))
        elif source in EXAMPLE_SOURCES:
            all_documents.extend(build_example_chunks(cleaned_content, base_metadata))
        elif source in OPTIONAL_RULE_SOURCES:
            if looks_like_rule_text(cleaned_content):
                all_documents.extend(build_rule_chunks(cleaned_content, base_metadata))

    return all_documents


def ingest_documents(documents: List[Document], reset_collection: bool = False) -> None:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if reset_collection:
        try:
            temp_store = PGVector(
                embeddings=embeddings,
                collection_name=COLLECTION_NAME,
                connection=CONNECTION_STRING,
                use_jsonb=True,
            )
            temp_store.delete_collection()
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    batch_size = 50
    print(f"Prepared {len(documents)} retrieval documents for ingestion into {COLLECTION_NAME}.")

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        vector_store.add_documents(batch)
        print(f"Loaded batch {start // batch_size + 1}")


if __name__ == "__main__":
    docs = build_documents(SOURCE_FILE)
    ingest_documents(docs, reset_collection=True)
