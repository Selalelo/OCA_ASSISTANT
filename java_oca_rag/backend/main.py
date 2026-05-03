import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from query import answer_question
from cache import get_cached, set_cache, get_cache_stats, is_redis_available
from auth import verify_token, save_chat_history, get_chat_history

load_dotenv()

app = FastAPI(
    title="Java OCA Study Assistant",
    description="RAG-powered study assistant for Java OCA SE 8",
    version="1.0.0"
)

# allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─── request models ───────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    filters: dict | None = None  # optional manual filters from frontend dropdowns


# ─── routes ───────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Java OCA Study Assistant API is running"}


@app.get("/health")
async def health():
    """System health check — shows cache stats and service status"""
    return {
        "status": "ok",
        "redis": is_redis_available(),
        "cache_stats": get_cache_stats()
    }


@app.post("/chat")
async def chat(request: ChatRequest, user: dict = Depends(verify_token)):
    """Main chat endpoint — protected, requires valid Supabase JWT"""
    
    user_id = user["id"]
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # step 1: check cache
    cached = get_cached(question)
    if cached:
        # save to history even if cached
        await save_chat_history(
            user_id=user_id,
            question=question,
            answer=cached["answer"],
            filters=cached["filters"],
            source="cache"
        )
        return {
            "answer": cached["answer"],
            "filters": cached["filters"],
            "chunks_retrieved": cached["chunks_retrieved"],
            "source": "cache"
        }

    # step 2: run full RAG pipeline
    result = answer_question(question)

    # step 3: cache the result
    set_cache(question, result)

    # step 4: save to supabase history
    await save_chat_history(
        user_id=user_id,
        question=question,
        answer=result["answer"],
        filters=result["filters"],
        source="rag"
    )

    return {
        "answer": result["answer"],
        "filters": result["filters"],
        "chunks_retrieved": result["chunks_retrieved"],
        "source": "rag"
    }


@app.get("/history")
async def history(user: dict = Depends(verify_token)):
    """Fetch user chat history from Supabase"""
    user_id = user["id"]
    records = await get_chat_history(user_id)
    return {"history": records}

# backend/main.py — add this endpoint
@app.get("/config")
async def get_config():
    return {
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY")
    }


@app.delete("/history")
async def clear_history(user: dict = Depends(verify_token)):
    """Clear user chat history — placeholder for now"""
    # can implement later with supabase delete call
    return {"message": "History cleared"}


# ─── run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)