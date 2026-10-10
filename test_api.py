import sys
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = sys.argv[1]

r = client.chat.completions.create(
    model=MODEL, temperature=0, max_tokens=200,
    messages=[{"role": "user", "content": "Say hello in one sentence."}])
print(r.choices[0].message.content)
