import os
import json
import fitz
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

from constants import TOPICS, DIFFICULTY, CONTENT_TYPES, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

DEFAULT_METADATA = {
    "topic": "unknown",
    "difficulty": "medium",
    "content_type": "concept",
    "exam_objective": "unknown"
}

FAILED_CHUNKS_FILE = "failed_chunks.json"
PROGRESS_SAVE_INTERVAL = 50  # save local progress every N chunks


# ---------------------------------------------------------------------------
# Progress persistence (local file — cheap fallback, no Qdrant roundtrip)
# ---------------------------------------------------------------------------

def save_progress(indexed_ids: set):
    """Persist indexed chunk IDs to disk so we can resume after a restart."""
    with open("indexing_progress.json", "w") as f:
        json.dump(list(indexed_ids), f)


def load_progress() -> set:
    """Load previously indexed chunk IDs from disk."""
    if os.path.exists("indexing_progress.json"):
        with open("indexing_progress.json", "r") as f:
            return set(json.load(f))
    return set()


def record_failed_chunk(chunk_id: int):
    """Append a failed chunk ID to the failures log for post-mortem inspection."""
    failures: list = []
    if os.path.exists(FAILED_CHUNKS_FILE):
        with open(FAILED_CHUNKS_FILE, "r") as f:
            try:
                failures = json.load(f)
            except json.JSONDecodeError:
                failures = []
    if chunk_id not in failures:
        failures.append(chunk_id)
    with open(FAILED_CHUNKS_FILE, "w") as f:
        json.dump(failures, f)


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def get_indexed_ids_from_qdrant() -> set:
    """Return the set of chunk IDs already present in Qdrant."""
    try:
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        return set(point.id for point in result[0])
    except Exception as e:
        # FIX: log the error instead of silently swallowing it.
        # Return an empty set so the caller can decide whether to abort or continue.
        print(f"Warning: could not fetch indexed IDs from Qdrant ({e}). "
              "Falling back to local progress file.")
        return set()


def create_collection():
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")

    for field in ["topic", "difficulty", "content_type"]:
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"Index ensured: {field}")
        except Exception as e:
            print(f"Index '{field}' already exists or failed: {e}")


# ---------------------------------------------------------------------------
# PDF extraction & chunking
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    print(f"Extracted {len(full_text)} characters from PDF")
    return full_text


def split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks, guarding against negative start index."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        # FIX: clamp to 0 so a CHUNK_OVERLAP larger than the remaining text
        # can never make `start` go negative and loop forever.
        start = max(0, end - CHUNK_OVERLAP)
        if end >= len(text):
            break
    print(f"Split into {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Metadata extraction (batched to reduce Groq API calls)
# ---------------------------------------------------------------------------

def _validate_metadata(parsed: dict) -> dict:
    """
    Ensure every metadata value is within the allowed sets defined in constants.
    Falls back to the default value for any field that is missing or out-of-range.
    FIX: previously only checked key presence, not value validity.
    """
    validated = {}
    allowed = {
        "topic": set(TOPICS),
        "difficulty": set(DIFFICULTY),
        "content_type": set(CONTENT_TYPES),
    }
    for key, default in DEFAULT_METADATA.items():
        value = parsed.get(key, default)
        if key in allowed and value not in allowed[key]:
            print(f"  Metadata warning: '{value}' is not a valid {key}, using default '{default}'")
            value = default
        validated[key] = value
    return validated


def extract_metadata_batch(chunks: list[str]) -> list[dict]:
    """
    Extract metadata for a batch of chunks in a single Groq call.
    FIX: batching 3–5 chunks per call dramatically reduces API usage vs one call per chunk.
    Returns a list of validated metadata dicts in the same order as the input.
    """
    numbered = "\n\n".join(
        f"[CHUNK {i}]\n{chunk[:400]}" for i, chunk in enumerate(chunks)
    )
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a metadata extractor. "
                        "Return a JSON array only — one object per chunk, in order. "
                        "No explanation. No backticks. No extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Extract structured fields for each Java OCA SE 8 chunk below.\n\n"
                        f"Rules:\n"
                        f"- topic must be one of: {TOPICS}\n"
                        f"- difficulty must be one of: {DIFFICULTY}\n"
                        f"- content_type must be one of: {CONTENT_TYPES}\n"
                        f"- exam_objective: short string like \"1.3\" or \"unknown\"\n\n"
                        f"{numbered}\n\n"
                        f'Output example (2 chunks): '
                        f'[{{"topic":"interfaces","content_type":"rule","difficulty":"medium","exam_objective":"1.4"}},'
                        f'{{"topic":"arrays","content_type":"concept","difficulty":"easy","exam_objective":"unknown"}}]'
                    ),
                },
            ],
        )

        raw = response.choices[0].message.content.strip()
        parsed_list = json.loads(raw)

        if not isinstance(parsed_list, list):
            parsed_list = [parsed_list]

        # Pad with defaults if the model returned fewer items than expected
        while len(parsed_list) < len(chunks):
            parsed_list.append(DEFAULT_METADATA.copy())

        return [_validate_metadata(item) for item in parsed_list[: len(chunks)]]

    except Exception as e:
        print(f"Warning: batch metadata extraction failed ({e}), using defaults for this batch")
        return [DEFAULT_METADATA.copy() for _ in chunks]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_chunk(chunk_id: int, chunk_text: str, metadata: dict):
    vector = embedder.encode(chunk_text).tolist()
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "text": chunk_text,
                    "topic": metadata["topic"],
                    "difficulty": metadata["difficulty"],
                    "content_type": metadata["content_type"],
                    "exam_objective": metadata["exam_objective"],
                },
            )
        ],
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

BATCH_SIZE = 5  # chunks per Groq metadata call


def process_pdf(pdf_path: str):
    print(f"\nStarting indexing: {pdf_path}")
    create_collection()

    # Prefer Qdrant as the source of truth; fall back to local file if unreachable.
    print("Checking existing progress in Qdrant...")
    already_indexed = get_indexed_ids_from_qdrant()
    if not already_indexed:
        already_indexed = load_progress()
        print(f"Using local progress file: {len(already_indexed)} chunks")
    else:
        print(f"Already indexed (Qdrant): {len(already_indexed)} chunks")

    full_text = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(full_text)
    total = len(chunks)

    # Collect chunks that still need processing
    pending = [(i, chunk) for i, chunk in enumerate(chunks) if i not in already_indexed]
    print(f"Chunks to process: {len(pending)} / {total}")

    indexed = 0
    locally_indexed: set = set(already_indexed)

    # Process in batches for efficient metadata extraction
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start: batch_start + BATCH_SIZE]
        batch_ids = [item[0] for item in batch]
        batch_texts = [item[1] for item in batch]

        print(f"\nBatch {batch_start // BATCH_SIZE + 1} — chunks {batch_ids[0]}–{batch_ids[-1]}")
        metadata_list = extract_metadata_batch(batch_texts)

        for (chunk_id, chunk_text), metadata in zip(batch, metadata_list):
            print(f"  Indexing chunk {chunk_id} | topic: {metadata['topic']} | "
                  f"difficulty: {metadata['difficulty']} | type: {metadata['content_type']}")

            success = False
            for attempt in range(3):
                try:
                    index_chunk(chunk_id, chunk_text, metadata)
                    success = True
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"    Retry {attempt + 1}/3 after error: {e}")
                        time.sleep(2)
                    else:
                        print(f"    Failed after 3 attempts — chunk {chunk_id} logged to {FAILED_CHUNKS_FILE}")
                        # FIX: persist failed IDs so you know what to re-run later
                        record_failed_chunk(chunk_id)

            if success:
                indexed += 1
                locally_indexed.add(chunk_id)

        # FIX: save local progress periodically so a mid-run crash loses minimal work
        if indexed > 0 and indexed % PROGRESS_SAVE_INTERVAL == 0:
            save_progress(locally_indexed)
            print(f"  Progress saved ({indexed} new chunks indexed so far)")

        # Small delay between batches to respect Groq rate limits
        time.sleep(0.5)

    # Final save
    save_progress(locally_indexed)
    skipped = total - len(pending)
    print(f"\nDone. {indexed} new chunks indexed. {skipped} skipped (already done).")
    print(f"Total in Qdrant: {len(locally_indexed)}")

    if os.path.exists(FAILED_CHUNKS_FILE):
        with open(FAILED_CHUNKS_FILE) as f:
            failures = json.load(f)
        if failures:
            print(f"Warning: {len(failures)} chunks failed permanently — see {FAILED_CHUNKS_FILE}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py path/to/your.pdf")
    else:
        process_pdf(sys.argv[1])