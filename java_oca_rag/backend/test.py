python -c "
import httpx
import os
from dotenv import load_dotenv
load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_ANON_KEY')

# test against the chat_history table directly
response = httpx.get(
    f'{url}/rest/v1/chat_history',
    headers={
        'apikey': key,
        'Authorization': f'Bearer {key}'
    }
)

print(f'Status: {response.status_code}')
print(f'Response: {response.text[:200]}')
"