"""LLM client abstraction with strategy pattern for different providers.

This module provides a unified interface for querying different LLM providers
(OpenAI, Groq, custom API gateways, etc.) with timing and token tracking.
"""
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class LLMResponse:
    """Standardized response from LLM query."""
    content: str
    raw_response: Any
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def query(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> LLMResponse:
        """Query the LLM with a prompt (and optional images) and return the response.

        Args:
            prompt: Text prompt.
            images: Optional list of image data URIs (e.g. "data:image/png;base64,...").
                Providers that do not support vision ignore this argument.
        """
        pass


def _build_multimodal_content(prompt: str, images: Optional[List[str]]):
    """Build an OpenAI-style message content payload.

    Returns a plain string when there are no images (preserving the text-only
    path exactly), or a list of content parts when images are supplied.
    """
    if not images:
        return prompt
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": img}})
    return content


class OpenAIDirectProvider(LLMProvider):
    """OpenAI direct API provider."""

    def __init__(self, model: str = "gpt-4o-2024-11-20", temperature: float = 1.0,
                 max_tokens: Optional[int] = None):
        import openai
        self.client = openai
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": _build_multimodal_content(prompt, images)}]
        create_kwargs = dict(model=self.model, messages=messages,
                             temperature=self.temperature, seed=42)
        if self.max_tokens:
            create_kwargs["max_tokens"] = self.max_tokens
        completion = self.client.chat.completions.create(**create_kwargs)

        # Extract token usage
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        try:
            usage = getattr(completion, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
        except Exception:
            pass

        return LLMResponse(
            content=completion.choices[0].message.content.strip(),
            raw_response=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class GroqProvider(LLMProvider):
    """Groq API provider."""

    def __init__(self, model: str = "llama3-8b-8192"):
        from groq import Groq
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def query(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model
        )
        return LLMResponse(
            content=response.choices[0].message.content.strip(),
            raw_response=response,
        )


def _is_reasoning_model(model: str) -> bool:
    """True for o-series (o1-/o3-/o4-*) and gpt-5* (except *chat*) models, which
    reject `temperature`/`max_tokens` and require `max_completion_tokens` (+ optional
    `reasoning_effort`)."""
    mdl = (model or "").lower()
    is_o_series = len(mdl) >= 2 and mdl[0] == "o" and mdl[1].isdigit()
    is_gpt5_reasoning = mdl.startswith("gpt-5") and "chat" not in mdl
    return is_o_series or is_gpt5_reasoning


class CustomAPIProvider(LLMProvider):
    """Custom OpenAI-compatible API gateway provider."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o-2024-11-20",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def query(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        reasoning = _is_reasoning_model(self.model)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _build_multimodal_content(prompt, images)}],
        }
        if reasoning:
            # Reasoning models: max_completion_tokens (+ reasoning_effort), NO temperature.
            payload["max_completion_tokens"] = self.max_tokens or 16384
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["temperature"] = self.temperature
            if self.max_tokens:
                payload["max_tokens"] = self.max_tokens

        timeout = 300 if reasoning else 120
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if 400 <= r.status_code < 500:
            # Diagnostic only: preserve the provider's error body (first 500
            # chars) so 4xx failures are attributable. No retry, no repair.
            raise requests.exceptions.HTTPError(
                f"{r.status_code} Client Error for {url}: {r.text[:500]}", response=r)
        r.raise_for_status()
        completion = r.json()

        # Extract token usage
        usage = completion.get("usage", {})

        return LLMResponse(
            content=completion["choices"][0]["message"]["content"].strip(),
            raw_response=completion,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )


def _data_uri_to_anthropic_image(uri: str) -> Dict[str, Any]:
    """Convert a `data:image/...;base64,<data>` URI to an Anthropic image block."""
    if not uri.startswith("data:"):
        raise ValueError(f"expected a data: URI, got {uri[:16]!r}...")
    header, _, data = uri.partition(",")
    media_type = header[len("data:"):].split(";", 1)[0] or "image/png"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


# A non-streaming POST returns no bytes until generation completes, so the
# request timeout is effectively a ceiling on total generation time. The frozen
# MIXED campaign caps output at 32000 (architecture) and 500 (selector), which
# always finishes well inside 300s. The uniform ALL_* tiers request the
# provider maximum (claude-opus-5: 128000), where a single generation can run
# far past 300s -- every retry then re-issues the identical request and the
# ladder walks 1/4 -> 4/4, killing the unit on infrastructure rather than
# science (mechanical ledger M5).
#
# Threshold is set at MIXED's own architecture ceiling so that every MIXED
# request keeps a bit-identical 300s timeout: this change is additive for the
# frozen campaign and only grants the longer horizon to high-ceiling tiers.
_TIMEOUT_STANDARD_S = 300
_TIMEOUT_HIGH_CEILING_S = 3600
_HIGH_CEILING_MAX_TOKENS = 32000


def request_timeout_s(max_tokens):
    """Seconds to wait on a non-streaming completion, by configured ceiling."""
    if not max_tokens or int(max_tokens) <= _HIGH_CEILING_MAX_TOKENS:
        return _TIMEOUT_STANDARD_S
    return _TIMEOUT_HIGH_CEILING_S



class AnthropicAPIProvider(LLMProvider):
    """Claude via the Anthropic Messages API, authenticated with ANTHROPIC_API_KEY.

    Public-release provider. The runs reported in the paper used a different
    authentication path that is not distributed here; see PROVENANCE.md, which
    also records the one difference this makes to what the model receives.

    SAMPLING PARAMETERS ARE MODEL-DEPENDENT. Some models reject `temperature`
    with a 400 whose message says "deprecated". Rather than hardcode a model
    list, this sends the parameter and, on that 400, strips it and retries once,
    caching the verdict per model for the process.
    """

    # model id -> False once the API has told us it rejects sampling params.
    _SAMPLING_OK: Dict[str, bool] = {}

    _API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "claude-sonnet-5", temperature: float = 1.0,
                 max_tokens: Optional[int] = None,
                 thinking: Optional[Dict[str, Any]] = None):
        self.model = model
        self.max_tokens = max_tokens or 4096
        self.temperature = temperature
        # Explicit reasoning/thinking control. Default None => the key is ABSENT
        # from the payload, i.e. exactly the historical behaviour, so every
        # existing caller stays byte-identical on the wire. V2 sets it explicitly.
        self.thinking = thinking

    def _api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running, e.g. "
                "`export ANTHROPIC_API_KEY=...`.")
        return key

    def query(self, prompt: str, images: Optional[List[str]] = None, **kwargs) -> LLMResponse:
        content: List[Dict[str, Any]] = []
        for img in images or []:
            content.append(_data_uri_to_anthropic_image(img))
        content.append({"type": "text", "text": prompt})

        headers = {
            "content-type": "application/json",
            "x-api-key": self._api_key(),
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        # Sent only when a caller states it explicitly; omitted otherwise so the
        # historical payload shape is unchanged.
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        # Send temperature unless this model has already told us it is deprecated.
        # Only sent when explicitly non-default: passing temperature=1.0 is a
        # no-op that would needlessly probe models which reject the parameter.
        _send_sampling = (self._SAMPLING_OK.get(self.model, True)
                          and self.temperature is not None
                          and abs(float(self.temperature) - 1.0) > 1e-9)
        if _send_sampling:
            payload["temperature"] = float(self.temperature)

        # Transient network/server failures must not kill a multi-hour run:
        # retry with exponential backoff. Only idempotent-safe failures are
        # retried (timeouts, connection resets, 429/5xx); auth/400s raise
        # immediately so real errors stay loud.
        completion = None
        last_exc = None
        for attempt in range(5):
            try:
                r = requests.post(self._API_URL, headers=headers, json=payload,
                                  timeout=request_timeout_s(self.max_tokens))
                # A model that deprecated sampling params answers 400 with an
                # explicit message. Strip them, remember it for this process, and
                # retry once — so the run continues at the API default rather
                # than dying, and later calls skip the doomed probe entirely.
                if (r.status_code == 400 and "temperature" in payload
                        and "deprecated" in r.text.lower()):
                    self._SAMPLING_OK[self.model] = False
                    payload.pop("temperature", None)
                    print(f"[llm] {self.model} rejects `temperature` "
                          f"(deprecated for this model); continuing at the API "
                          f"default. Sampling control needs a model that "
                          f"accepts it — haiku-4-5 does.", flush=True)
                    r = requests.post(self._API_URL, headers=headers,
                                      json=payload,
                                      timeout=request_timeout_s(self.max_tokens))
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"retryable status {r.status_code}: {r.text[:200]}")
                if 400 <= r.status_code < 500:
                    # Diagnostic only: preserve the provider's error body so a
                    # 4xx is attributable post-hoc (the 2026-08-20 Freeway
                    # smoke lost a 400's cause here). No retry, no repair.
                    raise requests.exceptions.HTTPError(
                        f"{r.status_code} Client Error for {self._API_URL}: {r.text[:500]}",
                        response=r)
                r.raise_for_status()
                completion = r.json()
                break
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError,
                    ValueError) as exc:
                last_exc = exc
                if isinstance(exc, requests.exceptions.HTTPError):
                    _status = getattr(getattr(exc, "response", None), "status_code", None)
                    if _status is not None and _status != 429 and _status < 500:
                        raise            # 4xx (auth, bad request) — not transient
                if attempt == 4:
                    raise
                delay = 5 * (2 ** attempt)      # 5, 10, 20, 40 s
                print(f"[llm] transient error ({type(exc).__name__}); retry "
                      f"{attempt + 1}/4 in {delay}s", flush=True)
                time.sleep(delay)
        if completion is None:                  # defensive: loop always breaks or raises
            raise last_exc or RuntimeError("LLM query failed")

        text = "".join(
            b.get("text", "")
            for b in completion.get("content", [])
            if b.get("type") == "text"
        ).strip()
        usage = completion.get("usage", {})
        # A truncated completion is NOT a model failure, but it looks exactly like
        # one downstream: the program is cut off mid-token, synthesis fails, and the
        # harness burns another call retrying. Measured on the archived runs, this
        # hit ONLY the dispatcher arm (lunarlander_disp rep0/1/2: 1/3/5 calls capped
        # at 16000, budget exhausted at 30/30 and 35/30) while its control never
        # exceeded 5,564 output tokens -- which confounds that ablation. Nothing in
        # the record distinguished the two cases, because stop_reason was discarded.
        # Print it loudly so a cap is never again mistaken for a bad model.
        if completion.get("stop_reason") == "max_tokens":
            print(f"[llm] TRUNCATED: hit max_tokens={self.max_tokens}. The program "
                  f"is incomplete; a synthesis failure here is a HARNESS CEILING, "
                  f"not a model error. Raise --max-tokens.", flush=True)
        return LLMResponse(
            content=text,
            raw_response=completion,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            total_tokens=(
                (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
            ) or None,
        )


class LangchainOpenAIProvider(LLMProvider):
    """LangChain OpenAI provider (legacy)."""

    def __init__(self, model: str = "gpt-4o-2024-11-20", temperature: float = 1.0):
        # Import only when needed to avoid hard dependency
        from langchain.prompts.chat import HumanMessagePromptTemplate
        # from langchain.chat_models import ChatOpenAI
        self.HumanMessagePromptTemplate = HumanMessagePromptTemplate
        self.model = model
        self.temperature = temperature
        self.client = None  # Will be initialized if needed

    def query(self, prompt: str, **kwargs) -> LLMResponse:
        chat_prompt = self.HumanMessagePromptTemplate.from_template(prompt)
        out = self.client.invoke(chat_prompt.to_messages())
        return LLMResponse(
            content=out.content,
            raw_response=None,
        )


class LLMClient:
    """Unified LLM client with timing and metrics tracking.

    This class provides a consistent interface for querying LLMs regardless
    of the underlying provider, while tracking timing and token usage.
    """

    def __init__(
        self,
        query_mode: str = "openai_direct",
        language_model: str = "gpt-4o-2024-11-20",
        temperature: float = 1.0,
        groq_model: str = "llama3-8b-8192",
        custom_base_url: Optional[str] = None,
        custom_api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        thinking: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the LLM client.

        Args:
            query_mode: Provider mode ('openai_direct', 'groq', 'custom', 'langchain_openai')
            language_model: Model name for OpenAI-style providers
            temperature: Sampling temperature
            groq_model: Model name for Groq provider
            custom_base_url: Base URL for custom API gateway
            custom_api_key: API key for custom gateway
        """
        self.query_mode = query_mode
        self.language_model = language_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking

        # Initialize timing
        self.timing: Dict[str, Any] = {"events": [], "rollup": {}}

        # Create the appropriate provider
        if query_mode == "openai_direct":
            self._provider = OpenAIDirectProvider(language_model, temperature, max_tokens)
        elif query_mode == "groq":
            self._provider = GroqProvider(groq_model)
        elif query_mode == "custom":
            if not custom_base_url:
                custom_base_url = os.environ.get("CUSTOM_BASE_URL")
            if not custom_api_key:
                custom_api_key = os.environ.get("CUSTOM_API_KEY")
            if not custom_base_url:
                raise ValueError("CUSTOM_BASE_URL must be set for query_mode='custom'")
            if not custom_api_key:
                raise ValueError("CUSTOM_API_KEY must be set for query_mode='custom'")
            self._provider = CustomAPIProvider(
                custom_base_url, custom_api_key, language_model, temperature, max_tokens,
                reasoning_effort,
            )
        elif query_mode == "anthropic":
            self._provider = AnthropicAPIProvider(language_model, temperature,
                                                  max_tokens, thinking)
        elif query_mode == "langchain_openai":
            self._provider = LangchainOpenAIProvider(language_model, temperature)
        else:
            raise ValueError(f"Unsupported query_mode: {query_mode}")

    def query(
        self,
        prompt: str,
        *,
        images: Optional[List[str]] = None,
        label: Optional[str] = None,
    ) -> Tuple[str, Any]:
        """Query the LLM with timing and metrics tracking.

        Args:
            prompt: The prompt to send to the LLM
            images: Optional list of image data URIs for multimodal (vision) queries.
                Ignored by text-only providers.
            label: Optional label for timing tracking

        Returns:
            Tuple of (response_content, raw_response)
        """
        meta_extra = {"model": self.language_model, "provider": self.query_mode}

        with self._record_time("llm", detail=label or self.query_mode, extra=meta_extra):
            response = self._provider.query(prompt, images=images)

        # Record token usage if available
        if response.total_tokens is not None:
            self.timing["events"][-1].update({
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            })

        # Carry the provider's stop reason onto the event so RunRecord.log_call can
        # persist it. Recorded independently of token usage: a truncated response
        # must stay visible even if a provider omits usage. Keys differ by
        # provider (Anthropic: stop_reason; OpenAI: choices[].finish_reason), and
        # both normalise to "max_tokens"/"length" respectively.
        _raw = response.raw_response if isinstance(response.raw_response, dict) else {}
        _stop = _raw.get("stop_reason")
        if _stop is None:
            try:
                _stop = (_raw.get("choices") or [{}])[0].get("finish_reason")
            except (AttributeError, IndexError, TypeError):
                _stop = None
        if _stop:
            self.timing["events"][-1]["stop_reason"] = \
                "max_tokens" if _stop == "length" else _stop

        return response.content, response.raw_response

    @contextmanager
    def _record_time(self, category: str, detail: Optional[str] = None, extra: Optional[dict] = None):
        """Context manager to capture wall time for any block."""
        t0 = time.perf_counter()
        err = None
        try:
            yield
        except Exception as e:
            err = str(e)
            raise
        finally:
            dt = time.perf_counter() - t0
            entry = {
                "ts": time.time(),
                "category": category,
                "detail": detail or "",
                "duration_s": dt,
            }
            if extra:
                entry.update(extra)
            if err:
                entry["error"] = err
            self.timing["events"].append(entry)

    def rollup_timings(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute rollup statistics from timing events."""
        out: Dict[Tuple[str, str], List[float]] = {}
        for ev in self.timing["events"]:
            key = (ev["category"], ev.get("detail", ""))
            out.setdefault(key, []).append(ev["duration_s"])

        roll: Dict[str, Dict[str, Dict[str, float]]] = {}
        for (cat, det), arr in out.items():
            roll.setdefault(cat, {})
            roll[cat][det] = {
                "count": len(arr),
                "total_s": sum(arr),
                "mean_s": mean(arr),
                "max_s": max(arr),
                "min_s": min(arr),
            }
        self.timing["rollup"] = roll
        return roll

    def init_timing(self):
        """Reset timing events and rollup."""
        self.timing = {"events": [], "rollup": {}}


def save_usage(client, path: str, **extra) -> dict:
    """Persist a completion/usage JSON for a run: totals (calls, tokens, LLM
    seconds), per-call events (with durations + labels), the rollup, plus any
    run-specific fields passed as kwargs (solved, level, wall_seconds, ...).

    Every runner should call this so accounting is uniform across the repo.
    """
    import json as _json
    import os as _os
    ev = client.timing.get("events", [])
    usage = {
        "model": getattr(client, "language_model", "?"),
        "query_mode": getattr(client, "query_mode", "?"),
        "totals": {
            "calls": len(ev),
            "tokens": sum((e.get("total_tokens") or 0) for e in ev),
            "prompt_tokens": sum((e.get("prompt_tokens") or 0) for e in ev),
            "completion_tokens": sum((e.get("completion_tokens") or 0) for e in ev),
            "llm_seconds": round(sum(e.get("duration_s", 0) for e in ev), 1),
        },
        **extra,
        "events": ev,
        "rollup": client.rollup_timings(),
    }
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(usage, f, indent=2, default=str)
    return usage


def load_prompt(name: str, game_name: Optional[str] = None, **kwargs) -> str:
    """Load a prompt from the abstraction_prompts directory.

    Looks for:
      1. abstraction_prompts/<game_name>/<name>.txt (game-specific)
      2. abstraction_prompts/<name>.txt (default)

    Args:
        name: Prompt file name (without .txt extension)
        game_name: Optional game name for game-specific prompts
        **kwargs: Format arguments for the prompt template

    Returns:
        Formatted prompt string

    Raises:
        FileNotFoundError: If prompt file not found
    """
    base = Path("abstraction_prompts")

    # Try game-specific prompt first
    if game_name:
        game_specific = base / game_name / f"{name}.txt"
        if game_specific.exists():
            text = game_specific.read_text()
            return text.format(**kwargs)

    # Fall back to default prompt
    default_path = base / f"{name}.txt"
    if default_path.exists():
        text = default_path.read_text()
        return text.format(**kwargs)

    raise FileNotFoundError(
        f"Prompt '{name}' not found in abstraction_prompts/ "
        + (f"or abstraction_prompts/{game_name}/" if game_name else "")
    )
