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
    
    # embed the question
    query_vector = embedder.encode(user_question).tolist()

    # build filter
    qdrant_filter = build_qdrant_filter(filters)

    # search
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=5
    )

    # extract text from top results
    chunks = [r.payload["text"] for r in results]
    
    print(f"Retrieved {len(chunks)} chunks")
    print(f"Filters applied: {filters}")
    
    return chunks


def generate_answer(user_question: str, chunks: list[str]) -> str:
    """Send retrieved chunks + question to LLM for final answer"""
    
    if not chunks:
        return "I could not find relevant information for your question. Try rephrasing or removing filters."

    context = "\n\n---\n\n".join(chunks)

    return llm_call(
        system="""You are a Java OCA SE 8 study assistant.
Answer questions using only the provided context.
If the answer is not in the context, say so clearly.
For practice questions, provide the answer and explain why.
For concepts, be clear and concise.
Use code examples where relevant.""",
        user=f"""Context:
{context}

Question: {user_question}""",
        max_tokens=1000
    )


def answer_question(user_question: str) -> dict:
    """Full query pipeline"""
    
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