import os
import json
import hashlib
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))

# in-memory fallback — resets on server restart, fine for dev
_memory_cache: dict = {}

redis_client = None

if REDIS_ENABLED:
    try:
        import redis
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            socket_connect_timeout=1,
            decode_responses=True
        )
        redis_client.ping()
        print("Cache: Redis connected")
    except Exception:
        redis_client = None
        print("Cache: Redis unavailable, falling back to in-memory")
else:
    print("Cache: Redis disabled, using in-memory")


def _make_key(question: str) -> str:
    """Normalize question and hash it for consistent cache keys"""
    normalized = question.lower().strip()
    return "oca:" + hashlib.md5(normalized.encode()).hexdigest()


def is_redis_available() -> bool:
    return redis_client is not None


def get_cached(question: str):
    key = _make_key(question)

    # try Redis first
    if redis_client:
        try:
            val = redis_client.get(key)
            if val:
                print(f"Cache hit (redis): {key}")
                return json.loads(val)
        except Exception:
            pass

    # fallback to memory
    entry = _memory_cache.get(key)
    if entry and time.time() < entry["expires"]:
        print(f"Cache hit (memory): {key}")
        return entry["data"]

    print(f"Cache miss: {key}")
    return None


def set_cache(question: str, value: dict):
    key = _make_key(question)

    # try Redis first
    if redis_client:
        try:
            redis_client.setex(key, CACHE_TTL, json.dumps(value))
            print(f"Cached in Redis: {key}")
            return
        except Exception:
            pass

    # fallback to memory
    _memory_cache[key] = {
        "data": value,
        "expires": time.time() + CACHE_TTL
    }
    print(f"Cached in memory: {key}")


def get_cache_stats():
    if redis_client:
        try:
            info = redis_client.info()
            return {
                "backend": "redis",
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0)
            }
        except Exception:
            pass
    return {
        "backend": "memory",
        "entries": len(_memory_cache),
        "active": sum(1 for e in _memory_cache.values() if time.time() < e["expires"])
    }