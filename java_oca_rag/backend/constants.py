# backend/constants.py
import os

# LLM config
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_MODEL = "llama-3.1-8b-instant"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
TOPICS = [
    "interfaces",
    "arrays",
    "exceptions",
    "strings",
    "OOP",
    "inheritance",
    "methods",
    "constructors",
    "data_types",
    "operators",
    "control_flow",
    "enums",
    "lambdas",
    "unknown"
]

DIFFICULTY = ["easy", "medium", "hard"]

CONTENT_TYPES = [
    "concept",
    "rule",
    "gotcha",
    "code_example",
    "practice_question"
]

COLLECTION_NAME = "java_oca"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CACHE_TTL = 3600  # 1 hour