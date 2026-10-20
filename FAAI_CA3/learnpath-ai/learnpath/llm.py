"""Optional LLM support. Mock mode (no LLM) is the default and needs no keys, downloads or internet.

Supported back-ends: Ollama (free, local) and the Anthropic API. Every LLM call goes through LLMHelper, which
counts calls and failures so the UI can show exactly what the model did, and every use has a rule-based fallback.
"""
import json
import re
import urllib.request

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def ollama_models(base_url=DEFAULT_OLLAMA_URL, timeout=2.0):
    """Return (ok, models_or_error). Lists locally installed Ollama models."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return True, sorted(m["name"] for m in data.get("models", []))
    except Exception as e:  # not running, wrong URL, timeout...
        return False, f"{type(e).__name__}: {e}"


def get_llm(mode, model="", base_url=DEFAULT_OLLAMA_URL):
    """Return a LangChain chat model, or None for mock mode."""
    if mode == "mock":
        return None
    from langchain.chat_models import init_chat_model

    if mode == "ollama":  # pip install langchain-ollama ; ollama pull llama3.1
        return init_chat_model(model or "llama3.1", model_provider="ollama", base_url=base_url, temperature=0)
    if mode == "anthropic":  # pip install langchain-anthropic ; needs ANTHROPIC_API_KEY
        return init_chat_model(model or "claude-sonnet-5-5", model_provider="anthropic")
    raise ValueError(f"Unknown mode {mode}")


def content_text(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def parse_json(text):
    """Extract the first JSON object or array from model output (handles code fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no valid JSON in model output")


class LLMHelper:
    """Thin wrapper around a LangChain chat model that counts calls and failures."""

    def __init__(self, llm, label):
        self.llm, self.label = llm, label
        self.calls, self.failures = 0, 0

    def text(self, system, user):
        self.calls += 1
        try:
            return content_text(self.llm.invoke([("system", system), ("human", user)]).content).strip()
        except Exception:
            self.failures += 1
            raise

    def structured(self, schema, system, user):
        """Structured output if the model supports it, otherwise JSON-in-text parsed and validated."""
        self.calls += 1
        try:
            return self.llm.with_structured_output(schema).invoke([("system", system), ("human", user)])
        except Exception:
            pass
        try:
            raw = content_text(self.llm.invoke([("system", system + " Reply with JSON only."), ("human", user)]).content)
            return schema.model_validate(parse_json(raw))
        except Exception:
            self.failures += 1
            raise
