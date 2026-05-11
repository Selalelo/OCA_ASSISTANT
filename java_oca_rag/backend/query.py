import os
import json
from groq import Groq
import anthropic
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from constants import (
    TOPICS, DIFFICULTY, CONTENT_TYPES,
    COLLECTION_NAME, LLM_PROVIDER,
    GROQ_MODEL, ANTHROPIC_MODEL
)

load_dotenv()

# clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def llm_call(system: str, user: str, max_tokens: int = 1000) -> str:
    """Single function that routes to Groq or Anthropic based on env"""
    if LLM_PROVIDER == "groq":
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return response.choices[0].message.content

    else:  # anthropic
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text


def is_casual_message(text: str) -> bool:
    """Quick heuristic to detect casual/social messages before hitting the LLM pipeline."""
    casual_patterns = [
        "thank", "thanks", "okay", "ok cool", "got it", "nice", "great",
        "awesome", "cool", "perfect", "understood", "makes sense", "cheers",
        "hi", "hello", "hey", "bye", "goodbye", "see you", "good morning",
        "good evening", "good night", "good luck", "you're welcome", "no problem",
        "np", "lol", "haha", "wow", "yes", "no", "sure", "alright"
    ]
    lower = text.strip().lower()
    # Short messages under 8 words that match casual patterns
    if len(lower.split()) <= 8:
        for pattern in casual_patterns:
            if pattern in lower:
                return True
    return False


def extract_query_filters(user_question: str) -> dict:
    """LLM extracts structured filters from natural language question"""
    result = llm_call(
        system="You are a filter extractor. Return JSON only. No explanation. No backticks.",
        user=f"""Extract search filters from this Java OCA SE 8 study question.

Rules:
- topic must be one of: {TOPICS} or null if not mentioned
- difficulty must be one of: {DIFFICULTY} or null if not mentioned
- content_type must be one of: {CONTENT_TYPES} or null if not mentioned

User question: {user_question}

Output example:
{{"topic": "interfaces", "difficulty": "hard", "content_type": "practice_question"}}
""",
        max_tokens=200
    )

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        print("Warning: bad JSON from filter extraction, using empty filters")
        return {"topic": None, "difficulty": None, "content_type": None}


def build_qdrant_filter(filters: dict):
    """Convert JSON filters into Qdrant filter object"""
    conditions = []

    if filters.get("topic"):
        conditions.append(
            FieldCondition(key="topic", match=MatchValue(value=filters["topic"]))
        )

    if filters.get("difficulty"):
        conditions.append(
            FieldCondition(key="difficulty", match=MatchValue(value=filters["difficulty"]))
        )

    if filters.get("content_type"):
        conditions.append(
            FieldCondition(key="content_type", match=MatchValue(value=filters["content_type"]))
        )

    return Filter(must=conditions) if conditions else None


def search_qdrant(user_question: str, filters: dict) -> list[str]:
    """Filter first, then rank by vector similarity"""

    query_vector = embedder.encode(user_question).tolist()
    qdrant_filter = build_qdrant_filter(filters)

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=5
    ).points

    chunks = [r.payload["text"] for r in results]

    print(f"Retrieved {len(chunks)} chunks")
    print(f"Filters applied: {filters}")

    return chunks


def generate_answer(user_question: str, chunks: list[str]) -> str:
    """Send retrieved chunks + question to LLM for final answer"""

    context = "\n\n---\n\n".join(chunks) if chunks else ""

    return llm_call(
        system="""You are a friendly Java OCA SE 8 study assistant.

First, decide what kind of message you're dealing with:

1. CASUAL / SOCIAL (greetings, thanks, "okay cool", "got it", "nice", "makes sense", etc.)
   → Reply naturally and briefly, like a helpful tutor would. Ignore the context.
   → Keep it warm and encouraging. 1-3 sentences max.
   → Examples: "Thanks!" → "You're welcome! Let me know if anything else comes up."
                "okay cool" → "Great! Feel free to ask whenever you're ready to dive into the next topic."

2. STUDY QUESTION (concepts, code, practice questions, explanations)
   → Use the provided context to answer clearly and concisely.
   → Do NOT reproduce raw quiz questions or answer lists from the context.
   → Explain concepts, give examples, and teach — don't dump source material.
   → If the context is empty or irrelevant, answer from your own Java OCA knowledge.

   STRICT CODE FORMATTING RULES for study answers:
   - Wrap ALL code in triple backticks with the java language tag
   - Every { must be followed by a newline and indented block
   - Every } must be on its own line
   - Every statement ending in ; must be on its own line
   - Nested blocks must be indented with 4 spaces
   - NEVER write multiple statements on the same line

   Correct example:
   ```java
   public class Dog extends Animal {
       @Override
       public void sound() {
           System.out.println("The dog barks.");
       }
   }
   ```""",
        user=f"""Context:
{context}

Message: {user_question}""",
        max_tokens=1000
    )


def answer_question(user_question: str) -> dict[str, str | dict | int]:
    """Full query pipeline"""

    # Short-circuit for casual messages — skip RAG entirely
    if is_casual_message(user_question):
        answer = generate_answer(user_question, [])
        return {
            "answer": answer,
            "filters": {"topic": None, "difficulty": None, "content_type": None},
            "chunks_retrieved": 0
        }

    # step 1: extract filters
    filters = extract_query_filters(user_question)

    # step 2: search qdrant
    chunks = search_qdrant(user_question, filters)

    # step 3: generate answer
    answer = generate_answer(user_question, chunks)

    return {
        "answer": answer,
        "filters": filters,
        "chunks_retrieved": len(chunks)
    }


# test directly
if __name__ == "__main__":
    question = "give me hard practice questions about interfaces"
    result = answer_question(question)
    print(f"\nFilters: {result['filters']}")
    print(f"Chunks retrieved: {result['chunks_retrieved']}")
    print(f"\nAnswer:\n{result['answer']}")