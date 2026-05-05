import os
import json
import fitz
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time
from qdrant_client.models import PayloadSchemaType

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

PROGRESS_FILE = "indexing_progress.json"


def save_progress(indexed_ids: set):
    """Save progress to file so we can resume after restart"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(indexed_ids), f)


def load_progress() -> set:
    """Load previously indexed chunk IDs"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    return set()


def get_indexed_ids_from_qdrant() -> set:
    """Check Qdrant directly for what's already indexed"""
    try:
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=False,
            with_vectors=False
        )
        return set(point.id for point in result[0])
    except:
        return set()


def create_collection():
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Collection '{COLLECTION_NAME}' created")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")

    # ensure payload indexes exist (idempotent — safe to run every time)
    for field in ["topic", "difficulty", "content_type"]:
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"Index ensured: {field}")
        except Exception as e:
            print(f"Index '{field}' already exists or failed: {e}")

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    print(f"Extracted {len(full_text)} characters from PDF")
    return full_text


def split_into_chunks(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    print(f"Split into {len(chunks)} chunks")
    return chunks


def extract_metadata(chunk_text: str) -> dict:
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": "You are a metadata extractor. Return a single JSON object only. No explanation. No backticks. No arrays."
                },
                {
                    "role": "user",
                    "content": f"""Extract structured fields from this Java OCA SE 8 study text.

Rules:
- topic must be one of: {TOPICS}
- difficulty must be one of: {DIFFICULTY}
- content_type must be one of: {CONTENT_TYPES}
- exam_objective: short string like "1.3" or "unknown"

Text:
{chunk_text}

Output example:
{{"topic": "interfaces", "content_type": "rule", "difficulty": "medium", "exam_objective": "1.4"}}
"""
                }
            ]
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else DEFAULT_METADATA

        if not isinstance(parsed, dict):
            return DEFAULT_METADATA

        for key in DEFAULT_METADATA:
            if key not in parsed or not parsed[key]:
                parsed[key] = DEFAULT_METADATA[key]

        return parsed

    except Exception as e:
        print(f"Warning: metadata extraction failed ({e}), using defaults")
        return DEFAULT_METADATA


def index_chunk(chunk_id: int, chunk_text: str, metadata: dict):
    vector = embedder.encode(chunk_text).tolist()
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=chunk_id,
            vector=vector,
            payload={
                "text": chunk_text,
                "topic": metadata["topic"],
                "difficulty": metadata["difficulty"],
                "content_type": metadata["content_type"],
                "exam_objective": metadata["exam_objective"]
            }
        )]
    )


def process_pdf(pdf_path: str):
    print(f"\nStarting indexing: {pdf_path}")
    create_collection()

    # check what's already indexed
    print("Checking existing progress in Qdrant...")
    already_indexed = get_indexed_ids_from_qdrant()
    print(f"Already indexed: {len(already_indexed)} chunks")

    full_text = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(full_text)
    total = len(chunks)

    skipped = 0
    indexed = 0

    for i, chunk in enumerate(chunks):

        # skip already indexed chunks
        if i in already_indexed:
            skipped += 1
            continue

        print(f"Processing chunk {i+1}/{total} (indexed: {indexed}, skipped: {skipped})...")

        metadata = extract_metadata(chunk)

        # retry logic for rate limits
        for attempt in range(3):
            try:
                index_chunk(i, chunk, metadata)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Retry {attempt+1}/3 after error: {e}")
                    time.sleep(2)
                else:
                    print(f"  Failed after 3 attempts, skipping chunk {i}")

        indexed += 1
        print(f"  topic: {metadata['topic']} | difficulty: {metadata['difficulty']} | type: {metadata['content_type']}")

        # small delay to avoid Groq rate limits
        time.sleep(0.3)

    print(f"\nDone. {indexed} new chunks indexed. {skipped} skipped (already done).")
    print(f"Total in Qdrant: {len(already_indexed) + indexed}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py path/to/your.pdf")
    else:
        process_pdf(sys.argv[1])
