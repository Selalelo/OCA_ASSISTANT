import os
import httpx
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── SERVE FRONTEND ──────────────────────────────────────────
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/login")
async def serve_login():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/app")
async def serve_app():
    return FileResponse(str(FRONTEND_DIR / "dashboard.html"))


# ── REQUEST MODELS ──────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    filters: dict | None = None


# ── ROUTES ──────────────────────────────────────────────────
from fastapi.responses import FileResponse, RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/login")


@app.get("/config")
async def get_config():
    return {
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY")
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "redis": is_redis_available(),
        "cache_stats": get_cache_stats()
    }


@app.post("/chat")
async def chat(request: ChatRequest, user: dict = Depends(verify_token)):
    user_id = user["id"]
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # check cache
    cached = get_cached(question)
    if cached:
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

    # run RAG pipeline
    result = answer_question(question)

    # cache result
    set_cache(question, result)

    # save to supabase
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

@app.get("/profile")
async def get_profile(user: dict = Depends(verify_token)):
    user_id = user["id"]
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{os.getenv('SUPABASE_URL')}/rest/v1/profiles",
            headers={
                "apikey": os.getenv("SUPABASE_ANON_KEY"),
                "Authorization": f"Bearer {os.getenv('SUPABASE_ANON_KEY')}"
            },
            params={
                "id": f"eq.{user_id}",
                "limit": "1"
            }
        )
    profiles = response.json()
    return profiles[0] if profiles else {"full_name": user.get("email")}


@app.get("/history")
async def history(user: dict = Depends(verify_token)):
    user_id = user["id"]
    records = await get_chat_history(user_id)
    return {"history": records}


@app.delete("/history")
async def clear_history(user: dict = Depends(verify_token)):
    return {"message": "History cleared"}


# ── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)