import os
import json
import fitz
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
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

        # if LLM returns a list, take first item
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else DEFAULT_METADATA

        # if still not a dict, use defaults
        if not isinstance(parsed, dict):
            return DEFAULT_METADATA

        # fill in any missing keys with defaults
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
    full_text = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(full_text)

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        metadata = extract_metadata(chunk)
        index_chunk(i, chunk, metadata)
        print(f"  topic: {metadata['topic']} | difficulty: {metadata['difficulty']} | type: {metadata['content_type']}")

    print(f"\nDone. {len(chunks)} chunks indexed into Qdrant.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py path/to/your.pdf")
    else:
        process_pdf(sys.argv[1])
