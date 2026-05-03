import os
import httpx
from fastapi import HTTPException, Header
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")


async def verify_token(authorization: str = Header(...)) -> dict:
    """Verify Supabase JWT token from request header"""
    
    # check header format
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Expected: Bearer <token>"
        )

    token = authorization.split(" ")[1]

    # verify with supabase
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON_KEY
            }
        )

    # invalid token
    if response.status_code != 200:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again."
        )

    user = response.json()
    return user


async def save_chat_history(
    user_id: str,
    question: str,
    answer: str,
    filters: dict,
    source: str
) -> None:
    """Save a chat message to Supabase"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/chat_history",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            json={
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "filters": filters,
                "source": source
            }
        )

        if response.status_code not in (200, 201):
            # don't crash app if history save fails
            print(f"Warning: failed to save chat history: {response.text}")


async def get_chat_history(user_id: str, limit: int = 50) -> list:
    """Fetch user chat history from Supabase"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/chat_history",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}"
            },
            params={
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
                "limit": str(limit)
            }
        )

        if response.status_code != 200:
            print(f"Warning: failed to fetch chat history: {response.text}")
            return []

        return response.json()