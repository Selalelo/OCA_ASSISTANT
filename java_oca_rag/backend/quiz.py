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

    prompt = f"""You are a Java OCA SE 8 exam question generator.

Using the context below, generate exactly {num_questions} multiple choice questions.
{"Topic focus: " + topic if topic else "Cover a broad range of OCA SE 8 topics."}

STRICT OUTPUT FORMAT — return a JSON array only, no explanation, no backticks:
[
  {{
    "id": 1,
    "topic": "inheritance",
    "difficulty": "medium",
    "question": "What is the output of the following code?",
    "options": {{
      "A": "A",
      "B": "B",
      "C": "Compile error",
      "D": "Runtime error"
    }},
    "correct": "B",
    "explanation": "Because of runtime polymorphism..."
  }}
]

Rules:
- Each question must have exactly 4 options: A, B, C, D
- correct must be one of: A, B, C, D
- Mix difficulties: 30% easy, 50% medium, 20% hard
- Include code snippet questions where relevant
- explanation must clearly explain WHY the correct answer is right
- CRITICAL: every string value must be complete — never truncate mid-sentence
- Return ONLY the JSON array, nothing else

Context:
{context}
"""

    result = llm_call(
        system="You are a Java OCA SE 8 exam generator. Return a valid JSON array only. No backticks. No explanation.",
        user=prompt,
        max_tokens=4000
    )

    try:
        cleaned = _strip_fences(result)
        questions = json.loads(cleaned)
        if isinstance(questions, list):
            return questions
    except json.JSONDecodeError as e:
        print(f"Quiz JSON parse error: {e}")
        print(f"Raw output: {result[:500]}")

    return []


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