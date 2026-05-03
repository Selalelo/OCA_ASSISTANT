import os
import json
import hashlib
import redis
from dotenv import load_dotenv

load_dotenv()

# client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True  # returns strings not bytes
)


def make_cache_key(question: str) -> str:
    """Convert question into a consistent cache key"""
    # lowercase + strip whitespace so slight variations hit same cache
    normalized = question.lower().strip()
    return f"oca:{hashlib.md5(normalized.encode()).hexdigest()}"


def get_cached(question: str) -> dict | None:
    """Check cache for existing answer"""
    key = make_cache_key(question)
    
    try:
        cached = redis_client.get(key)
        if cached:
            print(f"Cache hit: {key}")
            return json.loads(cached)
        print(f"Cache miss: {key}")
        return None
    except redis.RedisError as e:
        # if redis is down, don't crash the app
        print(f"Redis error: {e}")
        return None


def set_cache(question: str, result: dict, ttl: int = 3600) -> None:
    """Store answer in cache with expiry"""
    key = make_cache_key(question)
    
    try:
        redis_client.setex(
            key,
            ttl,           # expires after 1 hour by default
            json.dumps(result)
        )
        print(f"Cached: {key} (TTL: {ttl}s)")
    except redis.RedisError as e:
        print(f"Redis error: {e}")


def delete_cached(question: str) -> None:
    """Manually invalidate a cache entry"""
    key = make_cache_key(question)
    redis_client.delete(key)
    print(f"Deleted cache: {key}")


def get_cache_stats() -> dict:
    """Return basic cache stats for monitoring"""
    try:
        info = redis_client.info()
        return {
            "connected": True,
            "used_memory": info["used_memory_human"],
            "total_keys": redis_client.dbsize(),
            "hits": info["keyspace_hits"],
            "misses": info["keyspace_misses"],
            "hit_rate": round(
                info["keyspace_hits"] /
                max(info["keyspace_hits"] + info["keyspace_misses"], 1) * 100,
                2
            )
        }
    except redis.RedisError as e:
        return {"connected": False, "error": str(e)}


def is_redis_available() -> bool:
    """Health check for Redis"""
    try:
        redis_client.ping()
        return True
    except redis.RedisError:
        return False