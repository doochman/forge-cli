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

"""Runtime support for LLM-backed forge copilot generation.

This module remains the public orchestration surface for the copilot flow.
Low-level prompt building and contract-shape helpers live in dedicated modules,
but are re-exported here for compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from fluid_build.cli._common import redact_secrets, resolve_provider_from_contract
from fluid_build.cli.forge_copilot_contract_helpers import (
    KNOWN_BUILD_ENGINES,
    PROVIDER_ENGINE_COMPATIBILITY,
    TEMPLATE_ALIASES,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    _build_semantics_from_interview_summary as _build_semantics_from_interview_summary_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    _coerce_string_list as _coerce_string_list_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    _normalize_consumes_for_generation as _normalize_consumes_for_generation_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    _normalize_interview_summary as _normalize_interview_summary_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    build_seed_contract as build_seed_contract_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    classify_generation_failure as classify_generation_failure_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    extract_json_object as extract_json_object_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    normalize_generation_payload as normalize_generation_payload_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    normalize_provider_name as normalize_provider_name_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    normalize_template_name as normalize_template_name_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    redact_secret_like_text as redact_secret_like_text_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    sanitize_additional_files as sanitize_additional_files_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    sanitize_name as sanitize_name_impl,
)
from fluid_build.cli.forge_copilot_contract_helpers import (
    validate_generated_result as validate_generated_result_impl,
)
from fluid_build.cli.forge_copilot_discovery import DiscoveryReport, discover_local_context
from fluid_build.cli.forge_copilot_llm_providers import (
    BUILTIN_LLM_PROVIDERS,
    AnthropicProvider,
    CopilotGenerationError,
    GeminiProvider,
    LlmConfig,
    LlmProvider,
    OllamaProvider,
    OpenAIProvider,
    call_llm,
    get_llm_provider,
    resolve_llm_config,
)
from fluid_build.cli.forge_copilot_memory import CopilotMemorySnapshot
from fluid_build.cli.forge_copilot_prompts import (
    build_clarification_system_prompt as build_clarification_system_prompt_impl,
)
from fluid_build.cli.forge_copilot_prompts import (
    build_clarification_user_prompt as build_clarification_user_prompt_impl,
)
from fluid_build.cli.forge_copilot_prompts import (
    build_system_prompt as build_system_prompt_impl,
)
from fluid_build.cli.forge_copilot_prompts import (
    build_user_prompt as build_user_prompt_impl,
)
from fluid_build.cli.forge_copilot_schema_inference import (
    map_inferred_type_to_contract_type as _map_inferred_type_to_contract_type,
)
from fluid_build.schema_manager import FluidSchemaManager
from fluid_build.util.contract import get_builds

LOG = logging.getLogger("fluid.cli.forge_copilot")
COPILOT_BUILTIN_PROVIDERS = ("local", "gcp", "aws", "snowflake")


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


def build_capability_matrix() -> Dict[str, Any]:
    """Describe the locally available templates, providers, and supported engines."""
    warnings: List[str] = []
    try:
        from fluid_build.forge.core.registry import provider_registry, template_registry
    except Exception as exc:  # pragma: no cover - defensive import guard
        warnings.append(
            "Copilot couldn't inspect the local provider registry "
            f"({exc}). Continuing with built-in provider defaults."
        )
        return {
            "providers": list(COPILOT_BUILTIN_PROVIDERS),
            "templates": {},
            "build_engines": sorted(KNOWN_BUILD_ENGINES),
            "provider_engine_compatibility": {
                provider: sorted(engines)
                for provider, engines in PROVIDER_ENGINE_COMPATIBILITY.items()
            },
            "warnings": warnings,
        }

    try:
        discovered_provider_names = list(provider_registry.list_available())
    except Exception as exc:
        discovered_provider_names = []
        warnings.append(
            "Copilot couldn't list local providers "
            f"({exc}). Continuing with built-in provider defaults."
        )

    verified_provider_names: List[str] = []
    for provider_name in discovered_provider_names:
        try:
            provider = provider_registry.get(provider_name)
        except Exception as exc:  # pragma: no cover - registry.get already swallows today
            provider = None
            if provider_name in COPILOT_BUILTIN_PROVIDERS:
                warnings.append(
                    f"Copilot couldn't inspect the {provider_name} provider ({exc}). "
                    "Continuing without blocking the run."
                )
        if provider is None:
            if provider_name in COPILOT_BUILTIN_PROVIDERS:
                warnings.append(
                    f"Copilot couldn't inspect the {provider_name} provider. "
                    "Continuing without blocking the run."
                )
            continue
        verified_provider_names.append(provider_name)

    if verified_provider_names:
        provider_names = [
            provider
            for provider in COPILOT_BUILTIN_PROVIDERS
            if provider in verified_provider_names
        ]
        provider_names.extend(
            sorted(
                provider for provider in verified_provider_names if provider not in provider_names
            )
        )
    else:
        provider_names = list(COPILOT_BUILTIN_PROVIDERS)
        warnings.append(
            "Copilot couldn't verify any local providers, so it's using built-in provider defaults "
            "for planning. You can still review or override the provider later."
        )

    try:
        template_names = template_registry.list_available()
    except Exception as exc:
        template_names = []
        warnings.append(
            "Copilot couldn't list local templates "
            f"({exc}). Continuing with built-in defaults where possible."
        )
    templates: Dict[str, Any] = {}

    for template_name in template_names:
        try:
            template = template_registry.get(template_name)
        except Exception as exc:  # pragma: no cover - registry.get already swallows today
            warnings.append(
                f"Copilot couldn't inspect template '{template_name}' ({exc}). Continuing with the remaining templates."
            )
            continue
        if not template:
            warnings.append(
                f"Copilot couldn't inspect template '{template_name}'. Continuing with the remaining templates."
            )
            continue
        try:
            metadata = template.get_metadata()
        except Exception as exc:
            warnings.append(
                f"Copilot couldn't read metadata for template '{template_name}' ({exc}). Continuing with the remaining templates."
            )
            continue
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
        "warnings": warnings,
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
            attempt_summaries.append(
                f"Attempt {report.attempt}: parse error - {report.parse_error}"
            )
        elif report.validation_errors:
            joined = "; ".join(report.validation_errors[:4])
            attempt_summaries.append(f"Attempt {report.attempt}: validation failed - {joined}")
    failure_class = classify_generation_failure(attempts)
    raise CopilotGenerationError(
        "copilot_generation_failed",
        "Forge copilot could not produce a valid contract after 3 attempts.",
        suggestions=[
            "Check your project_goal/data_sources context for clarity",
            "Verify the selected model supports structured JSON responses",
            "Inspect discovery inputs for unsupported or ambiguous sources",
            *attempt_summaries[:3],
        ],
        context={"failure_class": failure_class, "attempt_summaries": attempt_summaries[:3]},
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
            str(context.get("use_case_other", "")),
            str(context.get("data_sources", "")),
            " ".join(discovery_report.provider_hints),
        ]
    ).lower()
    explicit_provider = normalize_provider_name(context.get("provider") or "")
    available_providers = set(capability_matrix.get("providers") or [])
    fallback_provider = (
        "local"
        if "local" in available_providers
        else (
            sorted(available_providers)[0] if available_providers else COPILOT_BUILTIN_PROVIDERS[0]
        )
    )
    if explicit_provider in available_providers:
        provider = explicit_provider
        provider_source = "explicit_context"
        provider_reason = (
            f"Using explicit provider hint '{explicit_provider}' from the current run."
        )
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
        provider_reason = (
            "Using the current run context because it references AWS-oriented sources."
        )
    elif any(token in text for token in ("gcp", "bigquery", "dataform", "composer")):
        provider = "gcp"
        provider_source = "heuristic_context"
        provider_reason = (
            "Using the current run context because it references GCP-oriented sources."
        )
    else:
        provider = ""
        provider_source = ""
        provider_reason = ""
    if not provider and project_memory:
        preferred_provider = normalize_provider_name(project_memory.preferred_provider)
        if preferred_provider in available_providers:
            provider = preferred_provider
            provider_source = "project_memory"
            provider_reason = f"Reusing saved project memory provider '{preferred_provider}' because the current run was ambiguous."
        else:
            for memory_hint in project_memory.provider_hints:
                candidate = normalize_provider_name(memory_hint)
                if candidate in available_providers:
                    provider = candidate
                    provider_source = "project_memory"
                    provider_reason = f"Using saved project memory provider hint '{candidate}' because no stronger current signal was available."
                    break
    if not provider:
        provider = fallback_provider
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
        template_reason = (
            "Using the current run context because it looks like a machine-learning pipeline."
        )
    elif any(token in text for token in ("stream", "kafka", "real-time", "realtime")):
        template = "streaming"
        template_source = "heuristic_context"
        template_reason = (
            "Using the current run context because it looks like a streaming workload."
        )
    elif any(
        token in text
        for token in (
            "etl",
            "ingest",
            "cdc",
            "multi-source",
            "sync",
            "data_platform",
            "data platform",
            "data lake",
            "lakehouse",
        )
    ):
        template = "etl_pipeline"
        template_source = "heuristic_context"
        template_reason = (
            "Using the current run context because it looks like an ingestion or ETL workload."
        )
    elif any(token in text for token in ("analytics", "report", "dashboard", "bi", "metric")):
        template = "analytics"
        template_source = "heuristic_context"
        template_reason = (
            "Using the current run context because it looks like an analytics project."
        )
    elif project_memory and normalize_template_name(project_memory.preferred_template) in templates:
        template = normalize_template_name(project_memory.preferred_template)
        template_source = "project_memory"
        template_reason = f"Reusing saved project memory template '{template}' because the current run was ambiguous."
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


def _normalize_interview_summary(context: Mapping[str, Any]) -> Dict[str, Any]:
    return _normalize_interview_summary_impl(context)


def _coerce_string_list(value: Any) -> List[str]:
    return _coerce_string_list_impl(value)


def _normalize_consumes_for_generation(value: Any) -> List[Dict[str, str]]:
    return _normalize_consumes_for_generation_impl(value)


def _build_semantics_from_interview_summary(
    *,
    columns: List[Dict[str, Any]],
    interview_summary: Mapping[str, Any],
    expose_name: str,
    description: str,
) -> Dict[str, Any]:
    return _build_semantics_from_interview_summary_impl(
        columns=columns,
        interview_summary=interview_summary,
        expose_name=expose_name,
        description=description,
    )


def build_seed_contract(
    *,
    context: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    template_name: str,
    provider_name: str,
    project_memory: Optional[CopilotMemorySnapshot] = None,
) -> Dict[str, Any]:
    return build_seed_contract_impl(
        context=context,
        discovery_report=discovery_report,
        template_name=template_name,
        provider_name=provider_name,
        project_memory=project_memory,
        map_inferred_type_fn=_map_inferred_type_to_contract_type,
    )


def build_system_prompt(capability_matrix: Mapping[str, Any]) -> str:
    return build_system_prompt_impl(capability_matrix, sorted(KNOWN_BUILD_ENGINES))


def build_clarification_system_prompt(capability_matrix: Mapping[str, Any]) -> str:
    return build_clarification_system_prompt_impl(capability_matrix)


def build_clarification_user_prompt(
    *,
    interview_state: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    capability_matrix: Mapping[str, Any],
    project_memory: Optional[CopilotMemorySnapshot] = None,
    previous_failure: Sequence[str] | None = None,
) -> str:
    return build_clarification_user_prompt_impl(
        interview_state=interview_state,
        discovery_report=discovery_report,
        capability_matrix=capability_matrix,
        project_memory=project_memory,
        previous_failure=previous_failure,
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
    return build_user_prompt_impl(
        context=context,
        discovery_report=discovery_report,
        capability_matrix=capability_matrix,
        seed_contract=seed_contract,
        seed_template=seed_template,
        seed_provider=seed_provider,
        attempt_index=attempt_index,
        previous_errors=previous_errors,
        previous_payload=previous_payload,
        project_memory=project_memory,
    )


def classify_generation_failure(attempts: Sequence[GenerationAttemptReport]) -> str:
    return classify_generation_failure_impl(attempts)


def normalize_generation_payload(
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    capabilities: Mapping[str, Any],
    seed_template: str,
    seed_provider: str,
) -> Dict[str, Any]:
    try:
        return normalize_generation_payload_impl(
            payload,
            context=context,
            discovery_report=discovery_report,
            seed_template=seed_template,
            seed_provider=seed_provider,
            resolve_provider_from_contract_fn=resolve_provider_from_contract,
            get_builds_fn=get_builds,
        )
    except ValueError as exc:
        raise CopilotGenerationError(
            "copilot_contract_missing",
            str(exc),
            suggestions=["Ensure the selected model returns strict JSON objects"],
        ) from exc


def validate_generated_result(
    normalized: Mapping[str, Any],
    *,
    capabilities: Mapping[str, Any],
    logger: Optional[logging.Logger] = None,
) -> tuple[List[str], List[str]]:
    return validate_generated_result_impl(
        normalized,
        capabilities=capabilities,
        logger=logger or LOG,
        schema_manager_cls=FluidSchemaManager,
        resolve_provider_from_contract_fn=resolve_provider_from_contract,
        get_builds_fn=get_builds,
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    return extract_json_object_impl(text)


def sanitize_additional_files(value: Any) -> Dict[str, str]:
    return sanitize_additional_files_impl(value)


def normalize_template_name(value: Any) -> str:
    return normalize_template_name_impl(value)


def normalize_provider_name(value: Any) -> str:
    return normalize_provider_name_impl(value)


def sanitize_name(value: Any) -> str:
    return sanitize_name_impl(value)


def redact_secret_like_text(text: str) -> str:
    return redact_secret_like_text_impl(text, redact_secrets_fn=redact_secrets)
