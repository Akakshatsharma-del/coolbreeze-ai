"""
test_llm_gateway.py
Quick sanity check for the Groq -> Gemini fallback chain.
Run with: python test_llm_gateway.py
"""

import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dj_ai_employee_main.settings")
django.setup()

from support import llm_gateway

print("=" * 60)
print("TEST 1: Normal call (should use Groq)")
print("=" * 60)
response = llm_gateway.create(
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Give me 5 names of animals"}],
    tools=None,
)
print("stop_reason:", response.stop_reason)
print("content:", response.content[0].text)

print("\n" + "=" * 60)
print("TEST 2: Forced Gemini fallback (simulate Groq failure)")
print("=" * 60)
original_groq_client = llm_gateway.groq_client
llm_gateway.groq_client = None  # force Groq call to raise, triggering fallback

response = llm_gateway.create(
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Give me 5 names of fruits"}],
    tools=None,
)
print("stop_reason:", response.stop_reason)
print("content:", response.content[0].text)

llm_gateway.groq_client = original_groq_client  # restore

print("\n" + "=" * 60)
print("TEST 3: Tool use (Groq)")
print("=" * 60)
tools = [{
    "name": "get_weather",
    "description": "Get the weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]
response = llm_gateway.create(
    system="You are a helpful assistant. Use tools when relevant.",
    messages=[{"role": "user", "content": "What's the weather in Mumbai?"}],
    tools=tools,
)
print("stop_reason:", response.stop_reason)
for block in response.content:
    print("block type:", block.type, "| name:", block.name, "| input:", block.input)

print("\nAll tests completed.")