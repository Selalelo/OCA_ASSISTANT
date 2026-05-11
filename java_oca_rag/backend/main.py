import os
import httpx
import pathlib
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from query import answer_question
from cache import get_cached, set_cache, get_cache_stats, is_redis_available
from auth import verify_token, save_chat_history, get_chat_history
from quiz import generate_quiz, generate_challenge, evaluate_challenge
from analytics import aggregate_progress, generate_suggestions 

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

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
BASE_DIR = pathlib.Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ── REQUEST MODELS ──────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    filters: dict | None = None

class QuizRequest(BaseModel):
    topic: str | None = None
    num_questions: int = 20

class QuizResultRequest(BaseModel):
    topic: str | None = None
    score: int
    total: int
    wrong_topics: list[str]
    time_taken: int  # seconds

class ChallengeRequest(BaseModel):
    topic: str

class ChallengeSubmitRequest(BaseModel):
    challenge: dict
    user_code: str

class ProgressSaveRequest(BaseModel):
    session_type: str           # "chat" | "quiz" | "challenge"
    topic: str | None = None
    difficulty: str | None = None
    score: int | None = None
    total: int | None = None
    passed: bool | None = None
    time_spent_secs: int | None = None
    weak_topics: list[str] = []

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/login")
async def serve_login():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/app")
async def serve_app():
    return FileResponse(str(FRONTEND_DIR / "dashboard.html"))

@app.get("/quiz")
async def serve_quiz():
    return FileResponse(str(FRONTEND_DIR / "quiz.html"))

@app.get("/challenge")
async def serve_challenge():
    return FileResponse(str(FRONTEND_DIR / "challenge.html"))

@app.post("/progress/save")
async def save_progress(request: ProgressSaveRequest, user: dict = Depends(verify_token)):
    """Persist a single study session to user_progress table."""
    user_id = user["id"]
    supabase_url = os.getenv("SUPABASE_URL")
    service_key  = os.getenv("SUPABASE_SERVICE_KEY")
    anon_key     = os.getenv("SUPABASE_ANON_KEY")
 
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{supabase_url}/rest/v1/user_progress",
            headers={
                "apikey":        anon_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json={
                "user_id":         user_id,
                "session_type":    request.session_type,
                "topic":           request.topic,
                "difficulty":      request.difficulty,
                "score":           request.score,
                "total":           request.total,
                "passed":          request.passed,
                "time_spent_secs": request.time_spent_secs,
                "weak_topics":     request.weak_topics,
            }
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=500, detail=f"DB write failed: {resp.text}")
 
    return {"saved": True}
 
 
@app.get("/progress/analytics")
async def get_analytics(user: dict = Depends(verify_token)):
    """
    Fetch all user_progress rows, aggregate, and return analytics payload.
    Also generates/refreshes AI suggestions if they are older than 24h.
    """
    user_id      = user["id"]
    supabase_url = os.getenv("SUPABASE_URL")
    service_key  = os.getenv("SUPABASE_SERVICE_KEY")
    anon_key     = os.getenv("SUPABASE_ANON_KEY")
 
    async with httpx.AsyncClient() as client:
        # 1. Fetch all progress rows
        rows_resp = await client.get(
            f"{supabase_url}/rest/v1/user_progress",
            headers={
                "apikey":        anon_key,
                "Authorization": f"Bearer {service_key}",
            },
            params={
                "user_id": f"eq.{user_id}",
                "order":   "created_at.asc",
                "limit":   "1000",
            }
        )
        rows = rows_resp.json() if rows_resp.status_code == 200 else []
 
        # 2. Aggregate
        analytics = aggregate_progress(rows)
 
        # 3. Check if we have fresh suggestions (< 24h old)
        sug_resp = await client.get(
            f"{supabase_url}/rest/v1/user_suggestions",
            headers={
                "apikey":        anon_key,
                "Authorization": f"Bearer {service_key}",
            },
            params={
                "user_id": f"eq.{user_id}",
                "order":   "generated_at.desc",
                "limit":   "1",
            }
        )
        suggestions = []
        needs_refresh = True
 
        if sug_resp.status_code == 200 and sug_resp.json():
            latest = sug_resp.json()[0]
            from datetime import datetime, timezone, timedelta
            generated_at = datetime.fromisoformat(
                latest["generated_at"].replace("Z", "+00:00")
            )
            if datetime.now(timezone.utc) - generated_at < timedelta(hours=24):
                suggestions = latest["suggestions"]
                needs_refresh = False
 
        if needs_refresh and rows:
            suggestions = generate_suggestions(analytics)
            # Persist new suggestions
            await client.post(
                f"{supabase_url}/rest/v1/user_suggestions",
                headers={
                    "apikey":        anon_key,
                    "Authorization": f"Bearer {service_key}",
                    "Content-Type":  "application/json",
                    "Prefer":        "return=minimal",
                },
                json={
                    "user_id":     user_id,
                    "suggestions": suggestions,
                }
            )
 
    return {
        "analytics":   analytics,
        "suggestions": suggestions,
    }
 
 
@app.get("/progress/history")
async def get_progress_history(user: dict = Depends(verify_token)):
    """Return raw progress rows for charting (last 90 days)."""
    user_id      = user["id"]
    supabase_url = os.getenv("SUPABASE_URL")
    service_key  = os.getenv("SUPABASE_SERVICE_KEY")
    anon_key     = os.getenv("SUPABASE_ANON_KEY")
 
    from datetime import datetime, timezone, timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
 
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{supabase_url}/rest/v1/user_progress",
            headers={
                "apikey":        anon_key,
                "Authorization": f"Bearer {service_key}",
            },
            params={
                "user_id":    f"eq.{user_id}",
                "created_at": f"gte.{cutoff}",
                "order":      "created_at.asc",
                "limit":      "500",
            }
        )
        rows = resp.json() if resp.status_code == 200 else []
 
    return {"rows": rows}




# ── ROUTES ──────────────────────────────────────────────────
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

    cached = get_cached(question)
    if cached:
        await save_chat_history(
            user_id=user_id,
            question=question,
            answer=cached["answer"],
            filters=cached["filters"],
            source="cache"
        )
        return {**cached, "source": "cache"}

    result = answer_question(question)
    set_cache(question, result)

    await save_chat_history(
        user_id=user_id,
        question=question,
        answer=result["answer"],
        filters=result["filters"],
        source="rag"
    )

    return {**result, "source": "rag"}


@app.post("/quiz/generate")
async def quiz_generate(request: QuizRequest, user: dict = Depends(verify_token)):
    """Generate a 20-question MCQ quiz"""
    questions = generate_quiz(topic=request.topic, num_questions=request.num_questions)
    if not questions:
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions")
    return {"questions": questions, "total": len(questions)}


@app.post("/quiz/save")
async def quiz_save(request: QuizResultRequest, user: dict = Depends(verify_token)):
    """Save quiz result to Supabase"""
    user_id = user["id"]
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    anon_key = os.getenv("SUPABASE_ANON_KEY")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{supabase_url}/rest/v1/quiz_results",
            headers={
                "apikey": anon_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            json={
                "user_id": user_id,
                "topic": request.topic,
                "score": request.score,
                "total": request.total,
                "percentage": round((request.score / request.total) * 100),
                "wrong_topics": request.wrong_topics,
                "time_taken": request.time_taken
            }
        )
        if response.status_code not in (200, 201):
            print(f"Warning: failed to save quiz result: {response.text}")

    return {"saved": True}


@app.get("/quiz/history")
async def quiz_history(user: dict = Depends(verify_token)):
    """Get past quiz results"""
    user_id = user["id"]
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    anon_key = os.getenv("SUPABASE_ANON_KEY")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{supabase_url}/rest/v1/quiz_results",
            headers={
                "apikey": anon_key,
                "Authorization": f"Bearer {service_key}",
            },
            params={
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
                "limit": "10"
            }
        )
        if response.status_code == 200:
            return {"results": response.json()}
        return {"results": []}


@app.post("/challenge/generate")
async def challenge_generate(request: ChallengeRequest, user: dict = Depends(verify_token)):
    """Generate a coding challenge"""
    challenge = generate_challenge(topic=request.topic)
    return challenge


@app.post("/challenge/submit")
async def challenge_submit(request: ChallengeSubmitRequest, user: dict = Depends(verify_token)):
    """Evaluate submitted code"""
    result = evaluate_challenge(
        challenge=request.challenge,
        user_code=request.user_code
    )
    return result


@app.get("/profile")
async def get_profile(user: dict = Depends(verify_token)):
    user_id = user["id"]
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{os.getenv('SUPABASE_URL')}/rest/v1/profiles",
            headers={
                "apikey": os.getenv("SUPABASE_ANON_KEY"),
                "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_KEY')}"
            },
            params={"id": f"eq.{user_id}", "limit": "1"}
        )
    profiles = response.json()
    return profiles[0] if profiles else {"full_name": user.get("email")}


@app.get("/history")
async def history(user: dict = Depends(verify_token)):
    user_id = user["id"]
    records = await get_chat_history(user_id)
    print(f"History for {user_id}: {len(records)} records")
    return {"history": records}


@app.delete("/history")
async def clear_history(user: dict = Depends(verify_token)):
    return {"message": "History cleared"}


@app.get("/progress")
async def serve_progress():
    return FileResponse(str(FRONTEND_DIR / "progress.html"))

# ── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)