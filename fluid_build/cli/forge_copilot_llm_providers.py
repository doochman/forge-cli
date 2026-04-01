# Copyright 2024-2026 Agentics Transformation Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM provider adapters and configuration for the forge copilot."""

from __future__ import annotations

__all__ = [
    "CopilotGenerationError",
    "LlmConfig",
    "LlmProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "BUILTIN_LLM_PROVIDERS",
    "get_llm_provider",
    "normalize_llm_provider_name",
    "resolve_llm_config",
    "call_llm",
]

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import httpx

from fluid_build.cli._common import CLIError

LOG = logging.getLogger("fluid.cli.forge_copilot.llm")


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class CopilotGenerationError(CLIError):
    """Structured error for copilot generation failures."""

    def __init__(
        self,
        event: str,
        message: str,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        payload = {"message": message}
        if context:
            payload.update(context)
        super().__init__(1, event, payload)
        self.message = message
        self.suggestions = suggestions or []


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LlmConfig:
    """Resolved configuration for a provider-backed LLM call."""

    provider: str
    model: str
    endpoint: str
    api_key: Optional[str]
    timeout_seconds: int = 120

    @property
    def redacted_endpoint(self) -> str:
        endpoint = self.endpoint
        endpoint = re.sub(r"([?&](?:key|token|api_key)=)[^&]+", r"\1***", endpoint, flags=re.I)
        return endpoint


# ---------------------------------------------------------------------------
# Provider Interface & Implementations
# ---------------------------------------------------------------------------


class LlmProvider(ABC):
    """Interface for provider-specific request/response translation."""

    name: str
    default_model: str

    @abstractmethod
    def default_endpoint(self, model: str, env: Mapping[str, str]) -> str:
        """Return the provider's default endpoint for the resolved model."""

    @abstractmethod
    def build_request(
        self, config: LlmConfig, system_prompt: str, user_prompt: str
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Build request headers and JSON payload."""

    @abstractmethod
    def extract_text(self, response_json: Dict[str, Any]) -> str:
        """Extract free-form response text from the provider response."""


class OpenAIProvider(LlmProvider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def default_endpoint(self, model: str, env: Mapping[str, str]) -> str:
        return (
            env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            + "/chat/completions"
        )

    def build_request(
        self, config: LlmConfig, system_prompt: str, user_prompt: str
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        payload = {
            "model": config.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        return headers, payload

    def extract_text(self, response_json: Dict[str, Any]) -> str:
        return response_json["choices"][0]["message"]["content"]


class OllamaProvider(OpenAIProvider):
    name = "ollama"
    default_model = "llama3.1"

    def default_endpoint(self, model: str, env: Mapping[str, str]) -> str:
        host = env.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        return host + "/v1/chat/completions"

    def build_request(
        self, config: LlmConfig, system_prompt: str, user_prompt: str
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        headers, payload = super().build_request(config, system_prompt, user_prompt)
        headers.pop("Authorization", None)
        return headers, payload


class AnthropicProvider(LlmProvider):
    name = "anthropic"
    default_model = "claude-3-5-sonnet-latest"

    def default_endpoint(self, model: str, env: Mapping[str, str]) -> str:
        return "https://api.anthropic.com/v1/messages"

    def build_request(
        self, config: LlmConfig, system_prompt: str, user_prompt: str
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if config.api_key:
            headers["x-api-key"] = config.api_key
        payload = {
            "model": config.model,
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        return headers, payload

    def extract_text(self, response_json: Dict[str, Any]) -> str:
        content = response_json.get("content") or []
        for part in content:
            if part.get("type") == "text":
                return part.get("text", "")
        raise KeyError("Anthropic response did not contain a text block")


class GeminiProvider(LlmProvider):
    name = "gemini"
    default_model = "gemini-2.5-flash"

    def default_endpoint(self, model: str, env: Mapping[str, str]) -> str:
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def build_request(
        self, config: LlmConfig, system_prompt: str, user_prompt: str
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["x-goog-api-key"] = config.api_key
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        return headers, payload

    def extract_text(self, response_json: Dict[str, Any]) -> str:
        candidates = response_json.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                text = part.get("text")
                if text:
                    return text
        raise KeyError("Gemini response did not contain any text")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILTIN_LLM_PROVIDERS: Dict[str, LlmProvider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "claude": AnthropicProvider(),
    "gemini": GeminiProvider(),
    "ollama": OllamaProvider(),
}


def normalize_llm_provider_name(value: Any) -> str:
    """Normalize LLM provider aliases (openai, anthropic, gemini, ollama).

    Unlike ``normalize_provider_name`` in ``forge_copilot_runtime`` (which
    handles infrastructure providers like gcp/aws/local), this function
    understands LLM-specific aliases such as ``"claude"`` → ``"anthropic"``.
    """
    if value is None:
        return "openai"
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "claude":
        return "anthropic"
    return normalized


def get_llm_provider(name: str) -> LlmProvider:
    """Resolve a provider adapter by name."""
    normalized = (name or "").strip().lower()
    provider = BUILTIN_LLM_PROVIDERS.get(normalized)
    if not provider:
        raise CopilotGenerationError(
            "copilot_invalid_llm_provider",
            f"Unsupported LLM provider '{name}'.",
            suggestions=[
                "Choose one of: openai, anthropic, gemini, ollama",
                "Use --llm-provider or FLUID_LLM_PROVIDER to select a provider",
            ],
        )
    return provider


# ---------------------------------------------------------------------------
# Config Resolution
# ---------------------------------------------------------------------------


def resolve_llm_config(args: Any, environ: Optional[Mapping[str, str]] = None) -> LlmConfig:
    """Resolve provider, model, endpoint, and API key from flags and env vars."""
    env = dict(environ or os.environ)
    provider_name = (
        getattr(args, "llm_provider", None)
        or env.get("FLUID_LLM_PROVIDER")
        or _infer_provider_from_env(env)
        or "openai"
    )
    provider = get_llm_provider(provider_name)
    model = getattr(args, "llm_model", None) or env.get("FLUID_LLM_MODEL") or provider.default_model
    if not model:
        raise CopilotGenerationError(
            "copilot_missing_llm_model",
            "No LLM model was configured for forge copilot.",
            suggestions=[
                "Set FLUID_LLM_MODEL before running fluid forge --mode copilot",
                "Or pass --llm-model on the command line",
            ],
        )

    endpoint = getattr(args, "llm_endpoint", None) or env.get("FLUID_LLM_ENDPOINT")
    if not endpoint:
        endpoint = provider.default_endpoint(model, env)

    api_key = _resolve_api_key(provider.name, env)
    if provider.name != "ollama" and not api_key:
        raise CopilotGenerationError(
            "copilot_missing_llm_api_key",
            f"No API key was configured for the {provider.name} copilot adapter.",
            suggestions=[
                "Set FLUID_LLM_API_KEY or the provider-specific API key environment variable",
                "Examples: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY",
                "For local models, use --llm-provider ollama and optionally --llm-endpoint",
            ],
        )

    return LlmConfig(provider=provider.name, model=model, endpoint=endpoint, api_key=api_key)


# ---------------------------------------------------------------------------
# LLM Call with Retry
# ---------------------------------------------------------------------------

_TRANSIENT_STATUS_CODES = {429, 502, 503, 504}
_LLM_MAX_RETRIES = 2
_LLM_RETRY_BASE_SECONDS = 2.0


def call_llm(
    provider: LlmProvider,
    config: LlmConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call the configured provider and return free-form response text."""
    headers, payload = provider.build_request(config, system_prompt, user_prompt)
    last_exc: Optional[Exception] = None

    _LLM_REQUEST_SUGGESTIONS = [
        "Check the selected model and endpoint are correct",
        "Verify the API key environment variable is set",
        "Use --llm-endpoint only when you need to override the provider default",
    ]

    for attempt in range(_LLM_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=config.timeout_seconds) as client:
                response = client.post(config.endpoint, headers=headers, json=payload)
                response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code in _TRANSIENT_STATUS_CODES and attempt < _LLM_MAX_RETRIES:
                delay = _LLM_RETRY_BASE_SECONDS * (2**attempt)
                LOG.info(
                    "LLM request returned %s, retrying in %.1fs", exc.response.status_code, delay
                )
                time.sleep(delay)
                continue
            raise CopilotGenerationError(
                "copilot_llm_request_failed",
                f"LLM request failed for provider {config.provider}: {exc}",
                suggestions=_LLM_REQUEST_SUGGESTIONS,
            ) from exc
        except httpx.HTTPError as exc:
            raise CopilotGenerationError(
                "copilot_llm_network_error",
                f"LLM network error for provider {config.provider}: {exc}",
                suggestions=_LLM_REQUEST_SUGGESTIONS,
            ) from exc

    try:
        return provider.extract_text(response.json())
    except Exception as exc:  # noqa: BLE001
        raise CopilotGenerationError(
            "copilot_llm_response_invalid",
            f"LLM response from {config.provider} could not be parsed.",
            suggestions=[
                "Verify the selected model supports JSON-friendly instruction following",
                "Try a different --llm-model or --llm-provider",
            ],
        ) from exc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _infer_provider_from_env(env: Mapping[str, str]) -> Optional[str]:
    detected = []
    if env.get("OPENAI_API_KEY"):
        detected.append("openai")
    if env.get("ANTHROPIC_API_KEY"):
        detected.append("anthropic")
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        detected.append("gemini")
    if env.get("OLLAMA_HOST"):
        detected.append("ollama")
    if len(detected) == 1:
        return detected[0]
    return None


def _resolve_api_key(provider: str, env: Mapping[str, str]) -> Optional[str]:
    if env.get("FLUID_LLM_API_KEY"):
        return env["FLUID_LLM_API_KEY"]
    if provider == "openai":
        return env.get("OPENAI_API_KEY")
    if provider == "anthropic":
        return env.get("ANTHROPIC_API_KEY")
    if provider == "gemini":
        return env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
    return None
