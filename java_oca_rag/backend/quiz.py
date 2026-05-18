import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from query import llm_call, search_qdrant
from constants import TOPICS

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def generate_quiz(topic: str | None, num_questions: int = 20) -> list[dict]:
    search_query = f"practice questions {topic}" if topic else "Java OCA SE 8 practice questions"
    filters = {"topic": topic, "difficulty": None, "content_type": None}
    chunks = search_qdrant(search_query, filters)
    context = "\n\n---\n\n".join(chunks) if chunks else "Java OCA SE 8 certification topics"

    all_questions = []
    batch_size = 5  # smaller batches = less chance of truncation
    batches = (num_questions + batch_size - 1) // batch_size

    for batch_num in range(batches):
        start_id = batch_num * batch_size + 1
        end_id = min(start_id + batch_size - 1, num_questions)
        count = end_id - start_id + 1

        prompt = f"""Generate exactly {count} Java OCA SE 8 multiple choice questions.
{"Topic: " + topic if topic else "Cover broad OCA SE 8 topics."}
Number them {start_id} to {end_id}.

Return ONLY a JSON array, no explanation, no backticks:
[
  {{
    "id": {start_id},
    "topic": "topic name",
    "difficulty": "easy|medium|hard",
    "question": "question text — include Java code using ```java\\n...\\n``` when relevant",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "A",
    "explanation": "why the correct answer is right"
  }}
]

Rules:
- Exactly 4 options: A, B, C, D
- correct is one of: A, B, C, D
- At least {max(1, count // 2)} questions must include a Java code snippet in the question field
- Code snippets must use fenced blocks: ```java\\n<code here>\\n```
- Code must be properly indented, one statement per line
- All strings must be complete, never truncated
- ONLY the JSON array, nothing else
"""

        result = llm_call(
            system="You are a Java OCA SE 8 exam generator. Return a valid JSON array only. No backticks. No explanation.",
            user=prompt,
            max_tokens=2000  # 2000 tokens per 5 questions = plenty of room
        )

        try:
            cleaned = _strip_fences(result)
            batch = json.loads(cleaned)
            if isinstance(batch, list):
                all_questions.extend(batch)
        except json.JSONDecodeError as e:
            print(f"Batch {batch_num + 1} parse error: {e}")
            print(f"Raw output: {result[:300]}")
            continue  # skip bad batch, don't fail entirely

    return all_questions


def generate_challenge(topic: str) -> dict:
    search_query = f"code examples {topic} Java"
    filters = {"topic": topic, "difficulty": None, "content_type": "code_example"}
    chunks = search_qdrant(search_query, filters)
    context = "\n\n---\n\n".join(chunks) if chunks else f"Java OCA SE 8 {topic}"

    prompt = f"""Generate a Java coding challenge on the topic: {topic}

Return a JSON object only, no backticks, no explanation:
{{
  "title": "Short descriptive title",
  "topic": "{topic}",
  "difficulty": "medium",
  "description": "Clear problem description",
  "requirements": ["requirement 1", "requirement 2"],
  "starter_code": "public class Solution {{\\n    public static void main(String[] args) {{\\n    }}\\n}}",
  "hints": ["hint 1", "hint 2"],
  "example_input": "optional example input",
  "example_output": "expected output",
  "solution": "complete working solution code"
}}

Rules:
- starter_code must be valid Java
- solution must be complete and correct
- Return ONLY the JSON object, nothing else

Context:
{context}
"""

    result = llm_call(
        system="You are a Java coding challenge generator. Return valid JSON only. No backticks. No explanation.",
        user=prompt,
        max_tokens=2000
    )

    try:
        cleaned = _strip_fences(result)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Challenge JSON parse error: {e}")

    return {
        "title": f"Java {topic} Challenge",
        "topic": topic,
        "difficulty": "medium",
        "description": "Challenge generation failed. Please try again.",
        "requirements": [],
        "starter_code": "public class Solution {\n    public static void main(String[] args) {\n        // your code here\n    }\n}",
        "hints": [],
        "example_input": "",
        "example_output": "",
        "solution": ""
    }


def evaluate_challenge(challenge: dict, user_code: str) -> dict:
    result = llm_call(
        system="You are a Java code reviewer. Return JSON only. No backticks. No explanation.",
        user=f"""Evaluate this Java solution for the given challenge.

Challenge: {challenge['title']}
Description: {challenge['description']}
Requirements: {json.dumps(challenge['requirements'])}
Expected output: {challenge['example_output']}

User's code:
{user_code}

Return JSON:
{{
  "passed": true or false,
  "score": 0-100,
  "feedback": "overall feedback",
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "correct_solution": "show the ideal solution if they failed"
}}""",
        max_tokens=1000
    )

    try:
        cleaned = _strip_fences(result)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "passed": False,
            "score": 0,
            "feedback": result,
            "issues": [],
            "suggestions": [],
            "correct_solution": challenge.get("solution", "")
        }