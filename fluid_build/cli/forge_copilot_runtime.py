"""Runtime support for LLM-backed forge copilot generation."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import httpx
import yaml

from fluid_build.cli._common import CLIError, resolve_provider_from_contract
from fluid_build.cli.forge_copilot_memory import CopilotMemorySnapshot
from fluid_build.config import RUN_STATE_DIR
from fluid_build.schema_manager import FluidSchemaManager
from fluid_build.util.contract import get_builds

LOG = logging.getLogger("fluid.cli.forge_copilot")

MAX_DISCOVERY_FILES = 300
MAX_SQL_FILES = 25
MAX_READMES = 10
MAX_SAMPLE_FILES = 12
MAX_EXISTING_CONTRACTS = 12
MAX_SAMPLE_ROWS = 20
MAX_README_LINES = 80
DISCOVERABLE_SAMPLE_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet", ".pq", ".avro"}
RUN_STATE_PATH_PARTS = tuple(Path(RUN_STATE_DIR).parts)

IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
}

SAFE_ADDITIONAL_FILE_EXTENSIONS = {
    ".py",
    ".sql",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
}


class CopilotGenerationError(CLIError):
    """Structured error for copilot generation failures."""

    def __init__(self, event: str, message: str, suggestions: Optional[List[str]] = None):
        super().__init__(1, event, {"message": message})
        self.message = message
        self.suggestions = suggestions or []


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


@dataclass
class DiscoveryReport:
    """Metadata-only view of locally discovered assets."""

    workspace_roots: List[str]
    files_scanned: int = 0
    detected_sources: List[Dict[str, Any]] = field(default_factory=list)
    sql_files: List[Dict[str, Any]] = field(default_factory=list)
    dbt_projects: List[Dict[str, Any]] = field(default_factory=list)
    terraform_projects: List[Dict[str, Any]] = field(default_factory=list)
    readmes: List[Dict[str, Any]] = field(default_factory=list)
    existing_contracts: List[Dict[str, Any]] = field(default_factory=list)
    sample_files: List[Dict[str, Any]] = field(default_factory=list)
    provider_hints: List[str] = field(default_factory=list)
    build_constraints: List[str] = field(default_factory=list)
    discovery_warnings: List[str] = field(default_factory=list)

    def to_prompt_payload(self) -> Dict[str, Any]:
        """Return a metadata-only payload safe to share with the LLM."""
        return {
            "workspace_roots": self.workspace_roots,
            "files_scanned": self.files_scanned,
            "detected_sources": self.detected_sources,
            "sql_files": self.sql_files,
            "dbt_projects": self.dbt_projects,
            "terraform_projects": self.terraform_projects,
            "readmes": self.readmes,
            "existing_contracts": self.existing_contracts,
            "sample_files": self.sample_files,
            "provider_hints": self.provider_hints,
            "build_constraints": self.build_constraints,
            "discovery_warnings": self.discovery_warnings,
        }


@dataclass
class GenerationAttemptReport:
    """Diagnostic information for a single generation attempt."""

    attempt: int
    raw_provider: str
    raw_model: str
    parse_error: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)


@dataclass
class ScaffoldDecisionReport:
    """Explain how scaffold seed guidance was chosen before LLM generation."""

    template: str
    provider: str
    template_source: str
    provider_source: str
    template_reason: str
    provider_reason: str


@dataclass
class CopilotGenerationResult:
    """Validated artifacts produced by the LLM-backed copilot flow."""

    suggestions: Dict[str, Any]
    contract: Dict[str, Any]
    readme_markdown: str
    additional_files: Dict[str, str]
    discovery_report: DiscoveryReport
    attempt_reports: List[GenerationAttemptReport]
    scaffold_decision: Optional[ScaffoldDecisionReport] = None
    project_memory: Optional[CopilotMemorySnapshot] = None


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
        return env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

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
            "max_tokens": 4000,
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
    default_model = "gemini-1.5-pro"

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


BUILTIN_LLM_PROVIDERS: Dict[str, LlmProvider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "claude": AnthropicProvider(),
    "gemini": GeminiProvider(),
    "ollama": OllamaProvider(),
}

TEMPLATE_ALIASES = {
    "analytics-dashboard": "analytics",
    "analytics_dashboard": "analytics",
    "analytics_basic": "analytics",
    "analytics-basic": "analytics",
    "analytics": "analytics",
    "etl": "etl_pipeline",
    "etl-pipeline": "etl_pipeline",
    "etl_pipeline": "etl_pipeline",
    "ml-pipeline": "ml_pipeline",
    "ml_pipeline": "ml_pipeline",
    "ml": "ml_pipeline",
    "starter": "starter",
    "startertemplate": "starter",
    "streaming": "streaming",
    "streaming-pipeline": "streaming",
    "streaming_pipeline": "streaming",
}

KNOWN_BUILD_ENGINES = {
    "sql",
    "python",
    "dbt",
    "dbt-bigquery",
    "dbt-athena",
    "dbt-redshift",
    "dbt-snowflake",
    "dataform",
    "glue",
}

PROVIDER_ENGINE_COMPATIBILITY = {
    "local": {"sql", "python", "dbt"},
    "gcp": {"sql", "python", "dbt", "dbt-bigquery", "dataform"},
    "aws": {"sql", "python", "dbt", "dbt-athena", "dbt-redshift", "glue"},
    "snowflake": {"sql", "python", "dbt", "dbt-snowflake"},
}


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


def discover_local_context(
    discovery_path: Optional[str],
    *,
    discover: bool = True,
    workspace_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> DiscoveryReport:
    """Scan local workspace files and return a metadata-only discovery report."""
    root = (workspace_root or Path.cwd()).resolve()
    roots = [root]
    if discovery_path:
        extra = Path(discovery_path).expanduser().resolve()
        if not extra.exists():
            raise CopilotGenerationError(
                "copilot_discovery_path_missing",
                f"Discovery path does not exist: {extra}",
                suggestions=["Check the path passed to --discovery-path"],
            )
        if extra not in roots:
            roots.append(extra)

    report = DiscoveryReport(
        workspace_roots=[str(path) for path in roots],
        build_constraints=[
            "Discovery payload must exclude raw sample rows, full file contents, and credentials.",
            "Use only providers and templates supported by the local Forge registries.",
            "Prefer placeholder env vars for destination configuration instead of hard-coded secrets.",
        ],
    )

    if not discover:
        report.build_constraints.append("Discovery was disabled by the user.")
        return report

    seen_files: set[Path] = set()
    provider_counts: Counter[str] = Counter()
    detected_sources: List[Dict[str, Any]] = []

    for scan_root in roots:
        for path in _iter_candidate_files(scan_root):
            if path in seen_files:
                continue
            if _is_excluded_discovery_artifact(path):
                continue
            seen_files.add(path)
            report.files_scanned += 1
            suffix = path.suffix.lower()

            if path.name == "dbt_project.yml":
                project = _summarize_dbt_project(path)
                report.dbt_projects.append(project)
                provider_counts.update(project.get("provider_hints") or [])
                continue

            if suffix == ".tf":
                terraform = _summarize_terraform_file(path)
                report.terraform_projects.append(terraform)
                provider_counts.update(terraform.get("provider_hints") or [])
                continue

            if path.name.lower().startswith("readme") and len(report.readmes) < MAX_READMES:
                report.readmes.append(_summarize_readme(path))
                continue

            if path.name.endswith("contract.fluid.yaml") or path.name.endswith("contract.fluid.json"):
                if len(report.existing_contracts) < MAX_EXISTING_CONTRACTS:
                    summary = _summarize_existing_contract(path)
                    report.existing_contracts.append(summary)
                    provider_counts.update(summary.get("providers") or [])
                continue

            if suffix == ".sql" and len(report.sql_files) < MAX_SQL_FILES:
                report.sql_files.append(_summarize_sql_file(path))
                continue

            if suffix in DISCOVERABLE_SAMPLE_SUFFIXES and len(report.sample_files) < MAX_SAMPLE_FILES:
                sample = _summarize_sample_file(path)
                report.sample_files.append(sample)
                detected_sources.append(sample)
                provider_counts.update(sample.get("provider_hints") or [])
                report.discovery_warnings.extend(sample.get("warnings") or [])

            if report.files_scanned >= MAX_DISCOVERY_FILES:
                break
        if report.files_scanned >= MAX_DISCOVERY_FILES:
            break

    report.detected_sources = detected_sources[:MAX_SAMPLE_FILES]
    report.provider_hints = [name for name, _ in provider_counts.most_common()]

    if report.sql_files:
        report.build_constraints.append(
            "Existing SQL assets were found; reuse discovered source table names and output naming conventions where possible."
        )
    if any(sample.get("format") == "parquet" for sample in report.sample_files):
        report.build_constraints.append(
            "Parquet files were discovered; prefer discovered column names, logical types, and storage conventions instead of inventing schemas."
        )
    if any(sample.get("format") == "avro" for sample in report.sample_files):
        report.build_constraints.append(
            "Avro files were discovered; preserve the discovered record shape and union/logical-type intent when generating exposes and builds."
        )
    if report.existing_contracts:
        report.build_constraints.append(
            "Existing FLUID contracts were found; stay consistent with discovered contract naming and provider conventions."
        )

    if logger:
        logger.debug(
            "copilot_discovery_complete",
            extra={
                "files_scanned": report.files_scanned,
                "provider_hints": report.provider_hints,
            },
        )

    return report


def build_capability_matrix() -> Dict[str, Any]:
    """Describe the locally available templates, providers, and supported engines."""
    from fluid_build.forge.core.registry import provider_registry, template_registry

    provider_names = provider_registry.list_available()
    template_names = template_registry.list_available()
    templates: Dict[str, Any] = {}

    for template_name in template_names:
        template = template_registry.get(template_name)
        if not template:
            continue
        metadata = template.get_metadata()
        templates[template_name] = {
            "description": metadata.description,
            "provider_support": [p for p in metadata.provider_support if p in provider_names],
            "use_cases": metadata.use_cases,
            "technologies": metadata.technologies,
        }

    return {
        "providers": provider_names,
        "templates": templates,
        "build_engines": sorted(KNOWN_BUILD_ENGINES),
        "provider_engine_compatibility": {
            provider: sorted(engines) for provider, engines in PROVIDER_ENGINE_COMPATIBILITY.items()
        },
    }


def generate_copilot_artifacts(
    context: Mapping[str, Any],
    *,
    llm_config: LlmConfig,
    discovery_report: DiscoveryReport,
    project_memory: Optional[CopilotMemorySnapshot] = None,
    capability_matrix: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    max_attempts: int = 3,
) -> CopilotGenerationResult:
    """Generate and validate copilot artifacts with a repair loop."""
    capabilities = dict(capability_matrix or build_capability_matrix())
    provider_adapter = get_llm_provider(llm_config.provider)
    scaffold_decision = _build_scaffold_decision(
        context,
        discovery_report,
        capabilities,
        project_memory=project_memory,
    )
    suggested_template = scaffold_decision.template
    suggested_provider = scaffold_decision.provider
    seed_contract = build_seed_contract(
        context=context,
        discovery_report=discovery_report,
        template_name=suggested_template,
        provider_name=suggested_provider,
        project_memory=project_memory,
    )

    attempts: List[GenerationAttemptReport] = []
    previous_errors: List[str] = []
    previous_payload: Optional[Dict[str, Any]] = None

    for attempt_index in range(1, max_attempts + 1):
        system_prompt = build_system_prompt(capabilities)
        user_prompt = build_user_prompt(
            context=context,
            discovery_report=discovery_report,
            capability_matrix=capabilities,
            seed_contract=seed_contract,
            seed_template=suggested_template,
            seed_provider=suggested_provider,
            attempt_index=attempt_index,
            previous_errors=previous_errors,
            previous_payload=previous_payload,
            project_memory=project_memory,
        )

        report = GenerationAttemptReport(
            attempt=attempt_index,
            raw_provider=llm_config.provider,
            raw_model=llm_config.model,
        )
        attempts.append(report)

        raw_text = call_llm(provider_adapter, llm_config, system_prompt, user_prompt)
        try:
            payload = extract_json_object(raw_text)
        except ValueError as exc:
            report.parse_error = str(exc)
            previous_errors = [report.parse_error]
            previous_payload = {"raw_text": redact_secret_like_text(raw_text[:2000])}
            continue

        normalized = normalize_generation_payload(
            payload,
            context=context,
            discovery_report=discovery_report,
            capabilities=capabilities,
            seed_template=suggested_template,
            seed_provider=suggested_provider,
        )
        validation_errors, validation_warnings = validate_generated_result(
            normalized,
            capabilities=capabilities,
            logger=logger,
        )
        report.validation_errors = validation_errors
        report.validation_warnings = validation_warnings

        if not validation_errors:
            return CopilotGenerationResult(
                suggestions=normalized["suggestions"],
                contract=normalized["contract"],
                readme_markdown=normalized["readme_markdown"],
                additional_files=normalized["additional_files"],
                discovery_report=discovery_report,
                attempt_reports=attempts,
                scaffold_decision=scaffold_decision,
                project_memory=project_memory,
            )

        previous_errors = validation_errors
        previous_payload = payload

    attempt_summaries = []
    for report in attempts:
        if report.parse_error:
            attempt_summaries.append(f"Attempt {report.attempt}: parse error - {report.parse_error}")
        elif report.validation_errors:
            joined = "; ".join(report.validation_errors[:4])
            attempt_summaries.append(f"Attempt {report.attempt}: validation failed - {joined}")
    raise CopilotGenerationError(
        "copilot_generation_failed",
        "Forge copilot could not produce a valid contract after 3 attempts.",
        suggestions=[
            "Check your project_goal/data_sources context for clarity",
            "Verify the selected model supports structured JSON responses",
            "Inspect discovery inputs for unsupported or ambiguous sources",
            *attempt_summaries[:3],
        ],
    )


def suggest_scaffold(
    context: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    capability_matrix: Mapping[str, Any],
    *,
    project_memory: Optional[CopilotMemorySnapshot] = None,
) -> tuple[str, str]:
    """Heuristically choose valid scaffold defaults used only as LLM guidance."""
    decision = _build_scaffold_decision(
        context,
        discovery_report,
        capability_matrix,
        project_memory=project_memory,
    )
    return decision.template, decision.provider


def _build_scaffold_decision(
    context: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    capability_matrix: Mapping[str, Any],
    *,
    project_memory: Optional[CopilotMemorySnapshot] = None,
) -> ScaffoldDecisionReport:
    """Build explainable scaffold guidance before LLM generation."""
    text = " ".join(
        [
            str(context.get("project_goal", "")),
            str(context.get("use_case", "")),
            str(context.get("data_sources", "")),
            " ".join(discovery_report.provider_hints),
        ]
    ).lower()
    explicit_provider = normalize_provider_name(context.get("provider") or "")
    available_providers = set(capability_matrix.get("providers") or [])
    if explicit_provider in available_providers:
        provider = explicit_provider
        provider_source = "explicit_context"
        provider_reason = f"Using explicit provider hint '{explicit_provider}' from the current run."
    elif discovery_report.provider_hints:
        provider = ""
        provider_source = ""
        provider_reason = ""
        for hint in discovery_report.provider_hints:
            candidate = normalize_provider_name(hint)
            if candidate in available_providers:
                provider = candidate
                provider_source = "current_discovery"
                provider_reason = (
                    f"Using current discovery provider hint '{candidate}' from local assets."
                )
                break
    elif "snowflake" in text:
        provider = "snowflake"
        provider_source = "heuristic_context"
        provider_reason = "Using the current run context because it references Snowflake."
    elif any(token in text for token in ("aws", "s3", "redshift", "athena", "glue")):
        provider = "aws"
        provider_source = "heuristic_context"
        provider_reason = "Using the current run context because it references AWS-oriented sources."
    elif any(token in text for token in ("gcp", "bigquery", "dataform", "composer")):
        provider = "gcp"
        provider_source = "heuristic_context"
        provider_reason = "Using the current run context because it references GCP-oriented sources."
    else:
        provider = ""
        provider_source = ""
        provider_reason = ""
    if not provider and project_memory:
        preferred_provider = normalize_provider_name(project_memory.preferred_provider)
        if preferred_provider in available_providers:
            provider = preferred_provider
            provider_source = "project_memory"
            provider_reason = (
                f"Reusing saved project memory provider '{preferred_provider}' because the current run was ambiguous."
            )
        else:
            for memory_hint in project_memory.provider_hints:
                candidate = normalize_provider_name(memory_hint)
                if candidate in available_providers:
                    provider = candidate
                    provider_source = "project_memory"
                    provider_reason = (
                        f"Using saved project memory provider hint '{candidate}' because no stronger current signal was available."
                    )
                    break
    if not provider:
        provider = "local" if "local" in available_providers else sorted(available_providers)[0]
        provider_source = "default"
        provider_reason = f"Falling back to the safe default provider '{provider}'."

    explicit_template = context.get("template") or context.get("recommended_template")
    template = normalize_template_name(explicit_template) if explicit_template else ""
    templates = set((capability_matrix.get("templates") or {}).keys())
    if template in templates:
        template_source = "explicit_context"
        template_reason = f"Using explicit template hint '{template}' from the current run."
    elif any(token in text for token in ("ml", "machine learning", "feature store", "model")):
        template = "ml_pipeline"
        template_source = "heuristic_context"
        template_reason = "Using the current run context because it looks like a machine-learning pipeline."
    elif any(token in text for token in ("stream", "kafka", "real-time", "realtime")):
        template = "streaming"
        template_source = "heuristic_context"
        template_reason = "Using the current run context because it looks like a streaming workload."
    elif any(token in text for token in ("etl", "ingest", "cdc", "multi-source", "sync")):
        template = "etl_pipeline"
        template_source = "heuristic_context"
        template_reason = "Using the current run context because it looks like an ingestion or ETL workload."
    elif any(token in text for token in ("analytics", "report", "dashboard", "bi", "metric")):
        template = "analytics"
        template_source = "heuristic_context"
        template_reason = "Using the current run context because it looks like an analytics project."
    elif project_memory and normalize_template_name(project_memory.preferred_template) in templates:
        template = normalize_template_name(project_memory.preferred_template)
        template_source = "project_memory"
        template_reason = (
            f"Reusing saved project memory template '{template}' because the current run was ambiguous."
        )
    else:
        template = "starter"
        template_source = "default"
        template_reason = "Falling back to the safe default template 'starter'."

    if template not in templates:
        template = "starter"
        template_source = "default"
        template_reason = "Falling back to the safe default template 'starter'."
    return ScaffoldDecisionReport(
        template=template,
        provider=provider,
        template_source=template_source,
        provider_source=provider_source,
        template_reason=template_reason,
        provider_reason=provider_reason,
    )


def build_seed_contract(
    *,
    context: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    template_name: str,
    provider_name: str,
    project_memory: Optional[CopilotMemorySnapshot] = None,
) -> Dict[str, Any]:
    """Create a minimal valid 0.7.1 contract used as guidance for the LLM."""
    project_name = sanitize_name(context.get("project_goal") or "copilot-data-product")
    expose_name = f"{project_name}_output"
    columns = [{"name": "id", "type": "integer", "required": True}]
    if discovery_report.sample_files:
        first = discovery_report.sample_files[0]
        columns = []
        for column_name, column_type in (first.get("columns") or {}).items():
            columns.append(
                {
                    "name": column_name,
                    "type": _map_inferred_type_to_contract_type(column_type),
                    "required": False,
                }
            )
        if not columns:
            columns = [{"name": "id", "type": "integer", "required": True}]

    binding = _default_binding(provider_name, expose_name)
    return {
        "fluidVersion": "0.7.1",
        "kind": "DataProduct",
        "id": f"generated.{project_name}",
        "name": project_name.replace("-", " ").title(),
        "description": context.get("project_goal") or "AI-generated FLUID data product",
        "domain": context.get("domain")
        or (project_memory.preferred_domain if project_memory else None)
        or context.get("use_case")
        or "analytics",
        "metadata": {
            "layer": "Bronze",
            "owner": {
                "team": context.get("owner")
                or (project_memory.preferred_owner if project_memory else None)
                or context.get("team_size")
                or "data-team",
                "email": "data-team@example.com",
            },
            "template": template_name,
        },
        "consumes": [],
        "builds": [
            {
                "id": "main_build",
                "pattern": "embedded-logic",
                "engine": "sql",
                "properties": {"sql": "SELECT 1 AS id"},
                "execution": {
                    "trigger": {"type": "manual", "iterations": 1},
                    "runtime": {
                        "platform": provider_name,
                        "resources": {"cpu": "1", "memory": "2Gi"},
                    },
                },
            }
        ],
        "exposes": [
            {
                "exposeId": expose_name,
                "kind": "table",
                "binding": binding,
                "contract": {"schema": columns},
            }
        ],
        "quality": [
            {
                "dimension": "completeness",
                "assertion": f"{columns[0]['name']} IS NOT NULL",
                "severity": "error",
            }
        ],
    }


def build_system_prompt(capability_matrix: Mapping[str, Any]) -> str:
    """System prompt for structured FLUID contract generation."""
    return (
        "You are FLUID Forge Copilot. Generate a production-ready FLUID contract and README "
        "that only use locally supported templates, providers, and build engines.\n"
        "Return strict JSON only. Do not wrap the response in markdown fences.\n"
        "Never include secrets, access tokens, raw sample values, or verbatim file contents.\n"
        "Use FLUID 0.7.1 unless the seed contract requires otherwise.\n"
        "Treat project_memory as a soft preference layer only. Explicit user context and the current "
        "discovery report take precedence.\n"
        "The JSON object must contain keys: recommended_template, recommended_provider, "
        "recommended_patterns, architecture_suggestions, best_practices, technology_stack, "
        "description, domain, owner, readme_markdown, contract, additional_files.\n"
        "The contract value must be a JSON object that passes FLUID validation and uses one of "
        f"these providers: {', '.join(capability_matrix.get('providers') or [])}.\n"
        "Only use build engines from the provided capability matrix."
    )


def build_user_prompt(
    *,
    context: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    capability_matrix: Mapping[str, Any],
    seed_contract: Mapping[str, Any],
    seed_template: str,
    seed_provider: str,
    attempt_index: int,
    previous_errors: Sequence[str],
    previous_payload: Optional[Mapping[str, Any]],
    project_memory: Optional[CopilotMemorySnapshot] = None,
) -> str:
    """Build the attempt-specific user prompt."""
    prompt: Dict[str, Any] = {
        "attempt": attempt_index,
        "user_context": {
            "project_goal": context.get("project_goal"),
            "data_sources": context.get("data_sources"),
            "use_case": context.get("use_case"),
            "complexity": context.get("complexity"),
            "team_size": context.get("team_size"),
            "domain": context.get("domain"),
            "provider_hint": context.get("provider"),
        },
        "capability_matrix": capability_matrix,
        "discovery_report": discovery_report.to_prompt_payload(),
        "seed_template": seed_template,
        "seed_provider": seed_provider,
        "seed_contract": seed_contract,
        "response_requirements": {
            "metadata_only_discovery": True,
            "include_additional_files_only_if_needed": True,
            "use_placeholder_env_vars_for_credentials": True,
            "prefer_manual_trigger_for_execute_compatibility": True,
        },
    }
    if project_memory:
        prompt["project_memory"] = project_memory.to_prompt_payload()
    if previous_errors:
        prompt["repair_feedback"] = list(previous_errors)
    if previous_payload:
        prompt["previous_response_summary"] = {
            key: value
            for key, value in previous_payload.items()
            if key in {"recommended_template", "recommended_provider", "contract"}
        }
    return json.dumps(prompt, indent=2, sort_keys=True)


def call_llm(
    provider: LlmProvider,
    config: LlmConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call the configured provider and return free-form response text."""
    headers, payload = provider.build_request(config, system_prompt, user_prompt)
    try:
        with httpx.Client(timeout=config.timeout_seconds) as client:
            response = client.post(config.endpoint, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise CopilotGenerationError(
            "copilot_llm_request_failed",
            f"LLM request failed for provider {config.provider}: {exc}",
            suggestions=[
                "Check the selected model and endpoint are correct",
                "Verify the API key environment variable is set",
                "Use --llm-endpoint only when you need to override the provider default",
            ],
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


def normalize_generation_payload(
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    capabilities: Mapping[str, Any],
    seed_template: str,
    seed_provider: str,
) -> Dict[str, Any]:
    """Normalize flexible LLM output into the shape the copilot flow expects."""
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        contract_yaml = payload.get("contract_yaml")
        if isinstance(contract_yaml, str):
            contract = yaml.safe_load(contract_yaml)
    if not isinstance(contract, dict):
        raise CopilotGenerationError(
            "copilot_contract_missing",
            "The LLM response did not include a valid contract object.",
            suggestions=["Ensure the selected model returns strict JSON objects"],
        )

    readme_markdown = payload.get("readme_markdown")
    if not isinstance(readme_markdown, str) or not readme_markdown.strip():
        readme_markdown = _build_default_readme(
            contract=contract,
            context=context,
            discovery_report=discovery_report,
        )

    recommended_provider = normalize_provider_name(
        payload.get("recommended_provider")
        or resolve_provider_from_contract(contract)[0]
        or seed_provider
    )
    recommended_template = normalize_template_name(payload.get("recommended_template") or seed_template)
    additional_files = sanitize_additional_files(payload.get("additional_files"))

    suggestions = {
        "recommended_template": recommended_template,
        "recommended_provider": recommended_provider,
        "recommended_patterns": list(payload.get("recommended_patterns") or []),
        "architecture_suggestions": list(payload.get("architecture_suggestions") or []),
        "best_practices": list(payload.get("best_practices") or []),
        "technology_stack": list(payload.get("technology_stack") or []),
        "description": payload.get("description") or contract.get("description") or "",
        "domain": payload.get("domain") or contract.get("domain") or context.get("domain") or "",
        "owner": payload.get("owner") or contract.get("metadata", {}).get("owner", {}).get("team"),
        "discovery_summary": {
            "provider_hints": discovery_report.provider_hints,
            "source_count": len(discovery_report.detected_sources),
        },
    }

    return {
        "contract": contract,
        "readme_markdown": readme_markdown,
        "additional_files": additional_files,
        "suggestions": suggestions,
    }


def validate_generated_result(
    normalized: Mapping[str, Any],
    *,
    capabilities: Mapping[str, Any],
    logger: Optional[logging.Logger] = None,
) -> tuple[List[str], List[str]]:
    """Validate contract schema plus local provider/build sanity checks."""
    contract = normalized["contract"]
    suggestions = normalized["suggestions"]
    errors: List[str] = []
    warnings: List[str] = []

    schema_manager = FluidSchemaManager(logger=logger or LOG)
    validation = schema_manager.validate_contract(contract, strict=True, offline_only=True)
    errors.extend(validation.errors)
    warnings.extend(validation.warnings)

    providers = set(capabilities.get("providers") or [])
    templates = set((capabilities.get("templates") or {}).keys())

    recommended_provider = suggestions.get("recommended_provider")
    if recommended_provider not in providers:
        errors.append(f"Unsupported provider '{recommended_provider}'. Use one of {sorted(providers)}")

    recommended_template = suggestions.get("recommended_template")
    if recommended_template not in templates:
        errors.append(
            f"Unsupported template '{recommended_template}'. Use one of {sorted(templates)}"
        )

    contract_provider, _location = resolve_provider_from_contract(contract)
    if not contract_provider:
        errors.append("Contract must expose at least one provider binding or runtime platform.")
    elif recommended_provider and contract_provider != recommended_provider:
        errors.append(
            f"Recommended provider '{recommended_provider}' does not match contract provider '{contract_provider}'."
        )

    builds = get_builds(contract)
    if not builds:
        errors.append("Contract must include a build or builds section.")
    for index, build in enumerate(builds):
        build_id = build.get("id") or f"build_{index}"
        engine = str(build.get("engine") or build.get("transformation", {}).get("engine") or "").strip()
        if not engine:
            errors.append(f"Build '{build_id}' is missing an engine.")
            continue
        if engine not in KNOWN_BUILD_ENGINES:
            errors.append(f"Build '{build_id}' uses unsupported engine '{engine}'.")
            continue
        if contract_provider:
            supported = PROVIDER_ENGINE_COMPATIBILITY.get(contract_provider, set())
            if supported and engine not in supported:
                errors.append(
                    f"Build '{build_id}' uses engine '{engine}' which is not supported for provider '{contract_provider}'."
                )
        if engine == "python":
            repository = build.get("repository")
            model = (build.get("properties") or {}).get("model")
            if not repository or not model:
                errors.append(
                    f"Python build '{build_id}' must define repository and properties.model for execute compatibility."
                )
        if engine == "sql":
            properties = build.get("properties") or {}
            sql_text = properties.get("sql") or properties.get("sql_statements") or build.get("sql")
            if not sql_text:
                errors.append(f"SQL build '{build_id}' must include SQL in build.properties.sql or sql_statements.")

    exposes = contract.get("exposes") or []
    if not exposes:
        errors.append("Contract must include at least one expose.")
    for expose in exposes:
        if not expose.get("exposeId"):
            errors.append("Each expose must include exposeId.")
        binding = expose.get("binding") or {}
        if not binding.get("platform"):
            errors.append(f"Expose '{expose.get('exposeId', 'unknown')}' is missing binding.platform.")
        contract_section = expose.get("contract") or {}
        schema = contract_section.get("schema")
        if not schema:
            warnings.append(f"Expose '{expose.get('exposeId', 'unknown')}' does not define contract.schema.")

    return errors, warnings


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a provider response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped)
        stripped = re.sub(r"```$", "", stripped).strip()

    for candidate in (stripped, _slice_first_json_object(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Response did not contain a valid JSON object.")


def sanitize_additional_files(value: Any) -> Dict[str, str]:
    """Accept a bounded set of relative text files proposed by the model."""
    if not isinstance(value, Mapping):
        return {}

    sanitized: Dict[str, str] = {}
    for raw_path, raw_content in value.items():
        if not isinstance(raw_path, str) or not isinstance(raw_content, str):
            continue
        candidate = Path(raw_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            continue
        if candidate.suffix.lower() not in SAFE_ADDITIONAL_FILE_EXTENSIONS:
            continue
        sanitized[str(candidate)] = raw_content
    return sanitized


def normalize_template_name(value: Any) -> str:
    """Normalize template aliases to registered template names."""
    if value is None:
        return "starter"
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = TEMPLATE_ALIASES.get(normalized, normalized)
    return normalized


def normalize_provider_name(value: Any) -> str:
    """Normalize provider aliases."""
    if value is None:
        return "local"
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "claude":
        return "anthropic"
    return normalized


def sanitize_name(value: Any) -> str:
    """Create a filesystem-safe identifier."""
    text = str(value or "copilot-data-product").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "copilot-data-product"


def redact_secret_like_text(text: str) -> str:
    """Best-effort redaction for common secret patterns."""
    redacted = re.sub(r"(Bearer\s+)[^\s]+", r"\1***", text, flags=re.I)
    redacted = re.sub(r"(x-api-key[\"']?\s*:\s*[\"'])[^\"']+", r"\1***", redacted, flags=re.I)
    redacted = re.sub(r"(api[_-]?key[\"']?\s*[:=]\s*[\"'])[^\"']+", r"\1***", redacted, flags=re.I)
    return redacted


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


def _iter_candidate_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return

    if not root.is_dir():
        return

    stack = [root]
    yielded = 0
    while stack and yielded < MAX_DISCOVERY_FILES:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name in IGNORED_DIRECTORIES:
                continue
            if entry.is_dir():
                stack.append(entry)
                continue
            yielded += 1
            yield entry
            if yielded >= MAX_DISCOVERY_FILES:
                return


def _is_run_state_artifact(path: Path) -> bool:
    """Return True when a file lives under the repo-local runtime state directory."""
    parts = path.parts
    if len(parts) < len(RUN_STATE_PATH_PARTS):
        return False
    for index in range(len(parts) - len(RUN_STATE_PATH_PARTS) + 1):
        if tuple(parts[index : index + len(RUN_STATE_PATH_PARTS)]) == RUN_STATE_PATH_PARTS:
            return True
    return False


def _is_excluded_discovery_artifact(path: Path) -> bool:
    """Return True when a file should be skipped by generic local discovery."""
    if _is_run_state_artifact(path):
        return True
    return any(part.lower() == "airbyte" for part in path.parts)


def _summarize_dbt_project(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profile = data.get("profile")
    return {
        "path": str(path),
        "name": data.get("name"),
        "profile": profile,
        "model_paths": data.get("model-paths") or [],
        "provider_hints": _extract_provider_hints(" ".join([str(profile), str(data)])),
    }


def _summarize_terraform_file(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    resource_matches = re.findall(r'resource\s+"([^"]+)"\s+"([^"]+)"', content)
    return {
        "path": str(path),
        "resources": [
            {"type": resource_type, "name": name} for resource_type, name in resource_matches[:15]
        ],
        "provider_hints": _extract_provider_hints(content),
    }


def _summarize_readme(path: Path) -> Dict[str, Any]:
    headings: List[str] = []
    words = 0
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if index >= MAX_README_LINES:
            break
        if line.strip().startswith("#"):
            headings.append(line.lstrip("#").strip())
        words += len(line.split())
    return {"path": str(path), "headings": headings[:12], "word_count": words}


def _summarize_existing_contract(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        contract = json.loads(raw)
    else:
        contract = yaml.safe_load(raw)
    providers = []
    for expose in contract.get("exposes") or []:
        binding = expose.get("binding") or {}
        if binding.get("platform"):
            providers.append(binding["platform"])
    return {
        "path": str(path),
        "fluid_version": contract.get("fluidVersion"),
        "kind": contract.get("kind"),
        "id": contract.get("id"),
        "name": contract.get("name"),
        "providers": sorted(set(providers)),
        "build_ids": [build.get("id") for build in get_builds(contract)[:10] if build.get("id")],
        "expose_ids": [
            expose.get("exposeId")
            for expose in (contract.get("exposes") or [])[:10]
            if expose.get("exposeId")
        ],
    }


def _summarize_sql_file(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    table_refs = re.findall(
        r"\b(?:from|join|into|update|table)\s+([A-Za-z0-9_.`\"]+)",
        content,
        flags=re.IGNORECASE,
    )
    return {
        "path": str(path),
        "line_count": len(content.splitlines()),
        "referenced_tables": table_refs[:15],
    }


def _summarize_sample_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    columns: Dict[str, str] = {}
    sampled_rows = 0
    row_count: Optional[int] = None
    schema_source: Optional[str] = None
    warnings: List[str] = []
    if suffix == ".csv":
        with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            type_tracker: Dict[str, List[str]] = {}
            for row in reader:
                sampled_rows += 1
                for key, value in row.items():
                    if key is None:
                        continue
                    type_tracker.setdefault(key, []).append(_infer_scalar_type(value))
                if sampled_rows >= MAX_SAMPLE_ROWS:
                    break
            columns = {key: _merge_types(values) for key, values in type_tracker.items()}
    elif suffix in {".json", ".jsonl"}:
        rows = list(_load_json_rows(path))
        sampled_rows = len(rows)
        type_tracker = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for key, value in row.items():
                type_tracker.setdefault(key, []).append(_infer_python_type(value))
        columns = {key: _merge_types(values) for key, values in type_tracker.items()}
    elif suffix in {".parquet", ".pq"}:
        metadata = _read_parquet_metadata(path)
        columns = metadata.get("columns") or {}
        row_count = metadata.get("row_count")
        schema_source = metadata.get("schema_source")
        warnings = list(metadata.get("warnings") or [])
    elif suffix == ".avro":
        metadata = _read_avro_metadata(path)
        columns = metadata.get("columns") or {}
        row_count = metadata.get("row_count")
        schema_source = metadata.get("schema_source")
        warnings = list(metadata.get("warnings") or [])

    summary = {
        "path": str(path),
        "format": suffix.lstrip("."),
        "sampled_rows": sampled_rows,
        "columns": columns,
        "provider_hints": _extract_provider_hints(path.name),
    }
    if row_count is not None:
        summary["row_count"] = row_count
    if schema_source:
        summary["schema_source"] = schema_source
    if warnings:
        summary["warnings"] = warnings
    return summary


def _read_parquet_metadata(path: Path) -> Dict[str, Any]:
    for reader in (_read_parquet_metadata_pyarrow, _read_parquet_metadata_duckdb):
        try:
            metadata = reader(path)
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001
            return {
                "columns": {},
                "warnings": [f"Could not inspect Parquet schema for {path.name}: {exc}"],
            }
        if metadata.get("columns"):
            return metadata

    return {
        "columns": {},
        "warnings": [
            f"Parquet file {path.name} was discovered but schema extraction requires pyarrow or duckdb."
        ],
    }


def _read_parquet_metadata_pyarrow(path: Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    columns = {field.name: _infer_arrow_type(str(field.type)) for field in schema}
    row_count = parquet_file.metadata.num_rows if parquet_file.metadata else None
    return {
        "columns": columns,
        "row_count": row_count,
        "schema_source": "pyarrow",
    }


def _read_parquet_metadata_duckdb(path: Path) -> Dict[str, Any]:
    import duckdb

    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(path)],
        ).fetchall()
    finally:
        connection.close()
    columns = {
        str(row[0]): _infer_duckdb_type(str(row[1]))
        for row in rows
        if len(row) >= 2 and row[0] is not None
    }
    return {"columns": columns, "schema_source": "duckdb"}


def _read_avro_metadata(path: Path) -> Dict[str, Any]:
    for reader in (_read_avro_metadata_fastavro, _read_avro_metadata_avro):
        try:
            metadata = reader(path)
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001
            return {
                "columns": {},
                "warnings": [f"Could not inspect Avro schema for {path.name}: {exc}"],
            }
        if metadata.get("columns"):
            return metadata

    return {
        "columns": {},
        "warnings": [
            f"Avro file {path.name} was discovered but schema extraction requires fastavro or avro."
        ],
    }


def _read_avro_metadata_fastavro(path: Path) -> Dict[str, Any]:
    from fastavro import reader

    with path.open("rb") as handle:
        avro_reader = reader(handle)
        schema = avro_reader.writer_schema or {}
    return {
        "columns": _extract_avro_columns(schema),
        "schema_source": "fastavro",
        "row_count": None,
    }


def _read_avro_metadata_avro(path: Path) -> Dict[str, Any]:
    from avro.datafile import DataFileReader
    from avro.io import DatumReader

    with path.open("rb") as handle:
        reader = DataFileReader(handle, DatumReader())
        try:
            schema = json.loads(str(reader.datum_reader.writers_schema))
        finally:
            reader.close()
    return {
        "columns": _extract_avro_columns(schema),
        "schema_source": "avro",
        "row_count": None,
    }


def _load_json_rows(path: Path) -> Iterable[Any]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".jsonl":
        for line in content.splitlines():
            if not line.strip():
                continue
            yield json.loads(line)
        return

    parsed = json.loads(content)
    if isinstance(parsed, list):
        for item in parsed[:MAX_SAMPLE_ROWS]:
            yield item
        return
    if isinstance(parsed, dict):
        if all(isinstance(value, list) for value in parsed.values()):
            keys = list(parsed.keys())
            row_count = min(len(parsed[key]) for key in keys)
            for index in range(min(row_count, MAX_SAMPLE_ROWS)):
                yield {key: parsed[key][index] for key in keys}
            return
        yield parsed


def _extract_avro_columns(schema: Mapping[str, Any]) -> Dict[str, str]:
    fields = schema.get("fields") or []
    columns: Dict[str, str] = {}
    for field in fields:
        name = field.get("name")
        if not name:
            continue
        columns[str(name)] = _infer_avro_type(field.get("type"))
    return columns


def _infer_avro_type(type_spec: Any) -> str:
    if isinstance(type_spec, list):
        non_null = [candidate for candidate in type_spec if candidate != "null"]
        if not non_null:
            return "string"
        return _infer_avro_type(non_null[0])
    if isinstance(type_spec, str):
        lowered = type_spec.lower()
        if lowered in {"boolean"}:
            return "boolean"
        if lowered in {"int", "long"}:
            return "integer"
        if lowered in {"float", "double"}:
            return "number"
        if lowered in {"bytes", "string", "enum"}:
            return "string"
        if lowered in {"array"}:
            return "array"
        if lowered in {"map", "record"}:
            return "object"
        return "string"
    if isinstance(type_spec, Mapping):
        logical_type = str(type_spec.get("logicalType") or "").lower()
        if logical_type in {"date"}:
            return "date"
        if logical_type in {"timestamp-millis", "timestamp-micros", "local-timestamp-millis", "local-timestamp-micros"}:
            return "datetime"

        avro_type = type_spec.get("type")
        if avro_type == "array":
            return "array"
        if avro_type in {"map", "record"}:
            return "object"
        if avro_type == "enum":
            return "string"
        return _infer_avro_type(avro_type)
    return "string"


def _extract_provider_hints(text: str) -> List[str]:
    lowered = text.lower()
    hints = []
    if any(token in lowered for token in ("gcp", "bigquery", "composer", "dataform")):
        hints.append("gcp")
    if any(token in lowered for token in ("aws", "s3", "redshift", "athena", "glue")):
        hints.append("aws")
    if "snowflake" in lowered:
        hints.append("snowflake")
    if not hints and any(token in lowered for token in ("csv", "json", "local", "duckdb")):
        hints.append("local")
    return hints


def _infer_arrow_type(type_name: str) -> str:
    lowered = type_name.lower()
    if any(token in lowered for token in ("bool",)):
        return "boolean"
    if any(token in lowered for token in ("int", "uint")):
        return "integer"
    if any(token in lowered for token in ("float", "double", "decimal")):
        return "number"
    if "timestamp" in lowered:
        return "datetime"
    if "date" in lowered:
        return "date"
    if any(token in lowered for token in ("list", "large_list", "fixed_size_list")):
        return "array"
    if any(token in lowered for token in ("struct", "map")):
        return "object"
    return "string"


def _infer_duckdb_type(type_name: str) -> str:
    lowered = type_name.lower()
    if "bool" in lowered:
        return "boolean"
    if any(token in lowered for token in ("tinyint", "smallint", "integer", "bigint", "hugeint")):
        return "integer"
    if any(token in lowered for token in ("float", "double", "decimal", "real")):
        return "number"
    if "timestamp" in lowered:
        return "datetime"
    if lowered == "date":
        return "date"
    if lowered.endswith("[]") or "list" in lowered or lowered.startswith("array"):
        return "array"
    if "struct" in lowered or "map" in lowered:
        return "object"
    return "string"


def _infer_scalar_type(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value).strip()
    if text == "":
        return "null"
    if text.lower() in {"true", "false"}:
        return "boolean"
    if re.fullmatch(r"-?\d+", text):
        return "integer"
    if re.fullmatch(r"-?\d+\.\d+", text):
        return "number"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return "date"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-Zz]+", text):
        return "datetime"
    return "string"


def _infer_python_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return _infer_scalar_type(value)


def _merge_types(values: Sequence[str]) -> str:
    filtered = [value for value in values if value != "null"]
    if not filtered:
        return "string"
    most_common = Counter(filtered).most_common(1)[0][0]
    return most_common


def _map_inferred_type_to_contract_type(value: str) -> str:
    mapping = {
        "boolean": "boolean",
        "integer": "integer",
        "number": "number",
        "date": "date",
        "datetime": "timestamp",
        "array": "array",
        "object": "object",
        "string": "string",
    }
    return mapping.get(value, "string")


def _default_binding(provider_name: str, expose_name: str) -> Dict[str, Any]:
    if provider_name == "gcp":
        return {
            "platform": "gcp",
            "format": "bigquery_table",
            "location": {
                "project": "${FLUID_GCP_PROJECT}",
                "dataset": "analytics",
                "table": expose_name,
            },
        }
    if provider_name == "aws":
        return {
            "platform": "aws",
            "format": "parquet",
            "location": {
                "bucket": "${FLUID_AWS_BUCKET}",
                "key": f"runtime/out/{expose_name}.parquet",
                "region": "${AWS_REGION}",
            },
        }
    if provider_name == "snowflake":
        return {
            "platform": "snowflake",
            "format": "table",
            "location": {
                "database": "${SNOWFLAKE_DATABASE}",
                "schema": "ANALYTICS",
                "table": expose_name.upper(),
            },
        }
    return {
        "platform": "local",
        "format": "csv",
        "location": {"path": f"runtime/out/{expose_name}.csv"},
    }


def _build_default_readme(
    *, contract: Mapping[str, Any], context: Mapping[str, Any], discovery_report: DiscoveryReport
) -> str:
    name = contract.get("name") or context.get("project_goal") or "FLUID Data Product"
    provider, _location = resolve_provider_from_contract(dict(contract))
    lines = [
        f"# {name}",
        "",
        "AI-generated FLUID project scaffold.",
        "",
        "## Summary",
        "",
        f"- Provider: {provider or 'unknown'}",
        f"- Domain: {contract.get('domain', 'analytics')}",
        f"- Builds: {len(get_builds(dict(contract)))}",
        f"- Discovered sources: {len(discovery_report.detected_sources)}",
        "",
        "## Quick Start",
        "",
        "```bash",
        "fluid validate contract.fluid.yaml",
        "fluid plan contract.fluid.yaml --provider local --out runtime/plan.json",
        "```",
        "",
        "## Notes",
        "",
        "This README was generated from metadata-only discovery. Review placeholder provider values before deployment.",
        "",
    ]
    return "\n".join(lines)


def _slice_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
