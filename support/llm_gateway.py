"""
llm_gateway.py

Drop-in replacement for `client.messages.create(...)` (the Anthropic SDK call).
Tries Groq first, falls back to Gemini on ANY failure (rate limit, auth error,
network error, etc). Always tries free providers only — no Anthropic anywhere.

Returned objects mimic the shape of the Anthropic SDK's Message response
(.stop_reason, .content -> list of blocks with .type/.text/.name/.input/.id)
so the existing agent-loop code in agents.py needs almost no changes.
"""

import json
import logging

from django.conf import settings
from groq import Groq
from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

GROQ_MODEL = getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")

groq_client = Groq(api_key=settings.GROQ_API_KEY) if getattr(settings, "GROQ_API_KEY", None) else None
gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY) if getattr(settings, "GEMINI_API_KEY", None) else None


# ---------------------------------------------------------------------------
# Normalized response objects (mimic Anthropic SDK shape)
# ---------------------------------------------------------------------------

class ContentBlock:
    def __init__(self, type, text=None, name=None, input=None, id=None):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.id = id


class NormalizedResponse:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


def _block_get(block, key, default=None):
    """Read an attribute whether block is our ContentBlock or a plain dict
    (dicts show up because previously-stored tool_result messages are dicts)."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


# ---------------------------------------------------------------------------
# Tool schema conversion (Anthropic input_schema format -> provider format)
# ---------------------------------------------------------------------------

def _tools_to_openai(tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _tools_to_gemini(tools):
    return [
        {
            "function_declarations": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }
                for t in tools
            ]
        }
    ]


# ---------------------------------------------------------------------------
# Message history conversion
# ---------------------------------------------------------------------------

def _messages_to_openai(system, messages):
    openai_messages = [{"role": "system", "content": system}]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts = []
            tool_calls = []
            for block in content:
                btype = _block_get(block, "type")
                if btype == "text":
                    text = _block_get(block, "text")
                    if text:
                        text_parts.append(text)
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": _block_get(block, "id"),
                        "type": "function",
                        "function": {
                            "name": _block_get(block, "name"),
                            "arguments": json.dumps(_block_get(block, "input")),
                        },
                    })
            openai_msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                openai_msg["tool_calls"] = tool_calls
            openai_messages.append(openai_msg)

        else:  # role == "user", content is a list of tool_result dicts
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    })
                else:
                    text = block.get("text") if isinstance(block, dict) else str(block)
                    openai_messages.append({"role": "user", "content": text})

    return openai_messages


def _messages_to_gemini(messages):
    contents = []
    tool_id_to_name = {}

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        gemini_role = "model" if role == "assistant" else "user"

        if isinstance(content, str):
            contents.append(genai_types.Content(role=gemini_role, parts=[genai_types.Part(text=content)]))
            continue

        parts = []
        for block in content:
            if role == "assistant":
                btype = _block_get(block, "type")
                if btype == "text":
                    text = _block_get(block, "text")
                    if text:
                        parts.append(genai_types.Part(text=text))
                elif btype == "tool_use":
                    name = _block_get(block, "name")
                    tool_id = _block_get(block, "id")
                    tool_id_to_name[tool_id] = name
                    parts.append(genai_types.Part(
                        function_call=genai_types.FunctionCall(name=name, args=_block_get(block, "input"))
                    ))
            else:  # user turn
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    name = tool_id_to_name.get(block["tool_use_id"], "unknown_tool")
                    parts.append(genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=name, response={"result": block["content"]}
                        )
                    ))
                else:
                    text = block.get("text") if isinstance(block, dict) else str(block)
                    parts.append(genai_types.Part(text=text))

        if parts:
            contents.append(genai_types.Content(role=gemini_role, parts=parts))

    return contents


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

def _call_groq(system, messages, tools):
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY not configured")

    kwargs = {
        "model": GROQ_MODEL,
        "max_tokens": 1024,
        "messages": _messages_to_openai(system, messages),
    }
    if tools:
        kwargs["tools"] = _tools_to_openai(tools)
        kwargs["tool_choice"] = "auto"

    resp = groq_client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content_blocks = []

    if choice.message.content:
        content_blocks.append(ContentBlock(type="text", text=choice.message.content))

    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            content_blocks.append(ContentBlock(
                type="tool_use",
                name=tc.function.name,
                input=json.loads(tc.function.arguments),
                id=tc.id,
            ))
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"
        if not content_blocks:
            content_blocks.append(ContentBlock(type="text", text=""))

    return NormalizedResponse(stop_reason=stop_reason, content=content_blocks)


def _call_gemini(system, messages, tools):
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY not configured")

    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        tools=_tools_to_gemini(tools) if tools else None,
        max_output_tokens=1024,
    )

    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_messages_to_gemini(messages),
        config=config,
    )

    content_blocks = []
    stop_reason = "end_turn"

    candidate = resp.candidates[0]
    parts = candidate.content.parts or []
    for i, part in enumerate(parts):
        if getattr(part, "function_call", None):
            fc = part.function_call
            content_blocks.append(ContentBlock(
                type="tool_use",
                name=fc.name,
                input=dict(fc.args) if fc.args else {},
                id=f"gemini_call_{i}_{fc.name}",
            ))
            stop_reason = "tool_use"
        elif getattr(part, "text", None):
            content_blocks.append(ContentBlock(type="text", text=part.text))

    if not content_blocks:
        content_blocks.append(ContentBlock(type="text", text=""))

    return NormalizedResponse(stop_reason=stop_reason, content=content_blocks)


# ---------------------------------------------------------------------------
# Public entry point — call this instead of client.messages.create(...)
# ---------------------------------------------------------------------------

def create(system, messages, tools=None):
    errors = []

    try:
        return _call_groq(system, messages, tools)
    except Exception as e:
        logger.warning("Groq call failed, falling back to Gemini: %s", e)
        errors.append(f"Groq: {e}")

    for attempt in range(2):
        try:
            return _call_gemini(system, messages, tools)
        except Exception as e:
            logger.error("Gemini call failed (attempt %d): %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(2)
            else:
                errors.append(f"Gemini: {e}")

    raise RuntimeError(f"All free LLM providers failed. Details: {' | '.join(errors)}")