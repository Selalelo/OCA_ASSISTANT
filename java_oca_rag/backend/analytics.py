"""
analytics.py
Aggregates raw user_progress rows into meaningful insights,
and uses the LLM to generate personalised improvement suggestions.
"""

import os
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from query import llm_call


# ── CONSTANTS ────────────────────────────────────────────────────────────────

ALL_TOPICS = [
    "OOP", "inheritance", "interfaces", "exceptions", "strings",
    "arrays", "methods", "constructors", "data_types",
    "operators", "control_flow", "enums", "lambdas"
]

MASTERY_THRESHOLDS = {
    "mastered":    80,   # ≥ 80 % → mastered
    "developing":  50,   # 50–79 % → developing
    "struggling":  0,    # < 50 % → struggling
}


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _classify_mastery(score_pct: float) -> str:
    if score_pct >= MASTERY_THRESHOLDS["mastered"]:
        return "mastered"
    if score_pct >= MASTERY_THRESHOLDS["developing"]:
        return "developing"
    return "struggling"


def _week_key(dt_str: str) -> str:
    """Return ISO week string like '2025-W21' from a timestamp string."""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-W%W")
    except Exception:
        return "unknown"


# ── CORE AGGREGATION ─────────────────────────────────────────────────────────

def aggregate_progress(rows: list[dict]) -> dict:
    """
    Turn raw user_progress rows into a structured analytics payload.

    Returns
    -------
    {
        total_sessions: int,
        total_time_mins: float,
        overall_score: float,          # weighted average across all sessions
        topic_stats: {
            topic: {
                attempts: int,
                avg_score: float,
                best_score: float,
                mastery: str,          # mastered / developing / struggling
                trend: str,            # improving / declining / stable
                session_types: [str],
            }
        },
        weekly_activity: {
            "2025-W21": { sessions: int, avg_score: float }
        },
        session_type_breakdown: {
            chat: int, quiz: int, challenge: int
        },
        weak_topics: [str],            # topics consistently below 50 %
        strong_topics: [str],          # topics above 80 %
        streak_days: int,              # consecutive days with at least one session
        recent_trend: str,             # "improving" | "declining" | "stable"
        last_7_days_sessions: int,
    }
    """
    if not rows:
        return _empty_analytics()

    topic_buckets: dict[str, list[float]] = defaultdict(list)
    topic_types: dict[str, set] = defaultdict(set)
    weekly: dict[str, list[float]] = defaultdict(list)
    type_count: dict[str, int] = defaultdict(int)
    all_scores: list[float] = []
    total_time = 0

    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)
    last_7_count = 0

    # Collect dates for streak calculation
    session_dates: set[str] = set()

    for row in rows:
        session_type = row.get("session_type", "chat")
        topic = row.get("topic") or "general"
        score = row.get("score")
        total = row.get("total") or 100
        passed = row.get("passed")
        time_secs = row.get("time_spent_secs") or 0
        created_at = row.get("created_at", "")

        # Normalise score to 0-100
        if score is not None:
            if session_type == "quiz" and total:
                pct = round((score / total) * 100, 1)
            elif session_type == "challenge":
                pct = 100.0 if passed else float(score or 0)
            else:
                pct = float(score)

            topic_buckets[topic].append(pct)
            all_scores.append(pct)

            week = _week_key(created_at)
            weekly[week].append(pct)

        type_count[session_type] += 1
        topic_types[topic].add(session_type)
        total_time += time_secs

        # Date for streak
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            session_dates.add(dt.strftime("%Y-%m-%d"))
            if dt >= seven_days_ago:
                last_7_count += 1
        except Exception:
            pass

    # Topic stats
    topic_stats = {}
    for topic, scores in topic_buckets.items():
        avg = round(sum(scores) / len(scores), 1)
        best = max(scores)

        # Trend: compare first half vs second half
        mid = len(scores) // 2
        if mid >= 1:
            first_avg = sum(scores[:mid]) / mid
            second_avg = sum(scores[mid:]) / (len(scores) - mid)
            if second_avg - first_avg > 5:
                trend = "improving"
            elif first_avg - second_avg > 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        topic_stats[topic] = {
            "attempts":     len(scores),
            "avg_score":    avg,
            "best_score":   round(best, 1),
            "mastery":      _classify_mastery(avg),
            "trend":        trend,
            "session_types": list(topic_types[topic]),
        }

    # Weekly activity
    weekly_activity = {
        week: {
            "sessions": len(scores),
            "avg_score": round(sum(scores) / len(scores), 1),
        }
        for week, scores in weekly.items()
    }

    # Weak / strong topics
    weak_topics = [t for t, s in topic_stats.items() if s["mastery"] == "struggling"]
    strong_topics = [t for t, s in topic_stats.items() if s["mastery"] == "mastered"]

    # Overall score
    overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    # Streak
    streak = _calculate_streak(session_dates)

    # Recent trend (last 5 sessions vs previous 5)
    recent_scores = [r.get("score") for r in rows[-10:] if r.get("score") is not None]
    if len(recent_scores) >= 4:
        half = len(recent_scores) // 2
        if sum(recent_scores[half:]) / (len(recent_scores) - half) > sum(recent_scores[:half]) / half + 3:
            recent_trend = "improving"
        elif sum(recent_scores[:half]) / half > sum(recent_scores[half:]) / (len(recent_scores) - half) + 3:
            recent_trend = "declining"
        else:
            recent_trend = "stable"
    else:
        recent_trend = "stable"

    return {
        "total_sessions":           len(rows),
        "total_time_mins":          round(total_time / 60, 1),
        "overall_score":            overall,
        "topic_stats":              topic_stats,
        "weekly_activity":          weekly_activity,
        "session_type_breakdown":   dict(type_count),
        "weak_topics":              weak_topics,
        "strong_topics":            strong_topics,
        "streak_days":              streak,
        "recent_trend":             recent_trend,
        "last_7_days_sessions":     last_7_count,
        "untouched_topics":         [t for t in ALL_TOPICS if t not in topic_buckets],
    }


def _calculate_streak(dates: set[str]) -> int:
    """Count consecutive days ending today (or yesterday)."""
    if not dates:
        return 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    streak = 0
    current = datetime.now(timezone.utc)
    while True:
        day_str = current.strftime("%Y-%m-%d")
        if day_str in dates:
            streak += 1
            current -= timedelta(days=1)
        else:
            break
    return streak


def _empty_analytics() -> dict:
    return {
        "total_sessions": 0,
        "total_time_mins": 0,
        "overall_score": 0,
        "topic_stats": {},
        "weekly_activity": {},
        "session_type_breakdown": {},
        "weak_topics": [],
        "strong_topics": [],
        "streak_days": 0,
        "recent_trend": "stable",
        "last_7_days_sessions": 0,
        "untouched_topics": list(ALL_TOPICS),
    }


# ── AI SUGGESTIONS ───────────────────────────────────────────────────────────

def generate_suggestions(analytics: dict) -> list[dict]:
    """
    Use the LLM to produce 4-6 actionable, personalised improvement suggestions
    based on the aggregated analytics.

    Each suggestion:
    {
        "priority": "high" | "medium" | "low",
        "category": "topic" | "habit" | "strategy" | "review",
        "title": str,
        "detail": str,
        "action": str,          # concrete next step
        "topic": str | null,    # relevant topic if category == "topic"
    }
    """
    weak = analytics.get("weak_topics", [])
    untouched = analytics.get("untouched_topics", [])
    strong = analytics.get("strong_topics", [])
    overall = analytics.get("overall_score", 0)
    streak = analytics.get("streak_days", 0)
    trend = analytics.get("recent_trend", "stable")
    sessions = analytics.get("total_sessions", 0)
    topic_stats = analytics.get("topic_stats", {})

    declining = [t for t, s in topic_stats.items() if s.get("trend") == "declining"]

    summary = f"""
Student Java OCA SE 8 study analytics:
- Overall score: {overall}%
- Total sessions: {sessions}
- Current streak: {streak} days
- Recent performance trend: {trend}
- Weak topics (avg < 50%): {weak}
- Declining topics: {declining}
- Untouched topics: {untouched}
- Strong topics (avg ≥ 80%): {strong}
- Topic details: {json.dumps({k: {"avg": v["avg_score"], "attempts": v["attempts"], "mastery": v["mastery"]} for k, v in topic_stats.items()}, indent=2)}
"""

    result = llm_call(
        system=(
            "You are a Java OCA SE 8 exam coach. "
            "Return a JSON array ONLY. No explanation, no backticks, no extra text. "
            "Each element is an object with keys: priority, category, title, detail, action, topic."
        ),
        user=f"""Based on this student's analytics, generate 5 specific, actionable improvement suggestions.

{summary}

Rules:
- priority must be one of: high, medium, low
- category must be one of: topic, habit, strategy, review
- title: short (5-8 words)
- detail: 1-2 sentences explaining WHY this matters
- action: a concrete next step the student can do TODAY
- topic: relevant topic name or null

Prioritise: weak topics > declining topics > untouched topics > habits.
Output example:
[
  {{
    "priority": "high",
    "category": "topic",
    "title": "Drill exceptions handling rules",
    "detail": "Your exceptions score is 34%, well below the 65% pass threshold. This topic appears on ~15% of OCA exam questions.",
    "action": "Ask the assistant for 5 hard practice questions on exceptions, then review each answer carefully.",
    "topic": "exceptions"
  }}
]
""",
        max_tokens=1200
    )

    try:
        clean = result.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception as e:
        print(f"Warning: suggestions parse failed ({e})")
        return _fallback_suggestions(weak, untouched)


def _fallback_suggestions(weak: list, untouched: list) -> list[dict]:
    suggestions = []
    for topic in weak[:2]:
        suggestions.append({
            "priority": "high",
            "category": "topic",
            "title": f"Focus on {topic}",
            "detail": f"Your {topic} score is below 50%. This needs immediate attention.",
            "action": f"Ask: 'give me hard practice questions about {topic}'",
            "topic": topic,
        })
    for topic in untouched[:2]:
        suggestions.append({
            "priority": "medium",
            "category": "topic",
            "title": f"Start studying {topic}",
            "detail": f"You haven't touched {topic} yet. It appears on the OCA exam.",
            "action": f"Ask: 'explain {topic} concepts for OCA'",
            "topic": topic,
        })
    return suggestions