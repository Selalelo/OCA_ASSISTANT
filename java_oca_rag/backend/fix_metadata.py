# test what groq is actually returning
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    max_tokens=200,
    messages=[
        {
            "role": "system",
            "content": "You are a metadata extractor. Return JSON only. No explanation. No backticks."
        },
        {
            "role": "user",
            "content": "Extract fields. topic: one of [interfaces, arrays]. difficulty: one of [easy, medium, hard]. content_type: one of [concept, rule]. exam_objective: string. Text: This chapter covers Java basics. Output: {\"topic\": \"...\", \"content_type\": \"...\", \"difficulty\": \"...\", \"exam_objective\": \"...\"}"
        }
    ]
)

print(repr(response.choices[0].message.content))
