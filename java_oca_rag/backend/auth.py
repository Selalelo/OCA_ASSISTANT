import os
import httpx
from typing import cast
from fastapi import HTTPException, Header
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = cast(str, os.getenv("SUPABASE_URL"))
SUPABASE_ANON_KEY = cast(str, os.getenv("SUPABASE_ANON_KEY"))
SUPABASE_SERVICE_KEY = cast(str, os.getenv("SUPABASE_SERVICE_KEY"))

assert SUPABASE_URL is not None, "SUPABASE_URL environment variable is required"
assert SUPABASE_ANON_KEY is not None, "SUPABASE_ANON_KEY environment variable is required"
assert SUPABASE_SERVICE_KEY is not None, "SUPABASE_SERVICE_KEY environment variable is required"


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
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
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


async def get_chat_history(user_id: str) -> list:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/chat_history",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_KEY')}",
            },
            params={
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
                "limit": "20"
            }
        )
        print(f"History fetch status: {response.status_code}")
        print(f"History fetch response: {response.text[:300]}")  # first 300 chars
        if response.status_code == 200:
            return response.json()
        return []