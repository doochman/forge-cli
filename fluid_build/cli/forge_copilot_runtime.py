"""Runtime support for LLM-backed forge copilot generation.

This module is the **orchestration layer** for the copilot flow.  Low-level
concerns live in dedicated sub-modules:

* ``forge_copilot_llm_providers`` – LLM provider adapters, config resolution,
  and the ``call_llm`` function.
* ``forge_copilot_discovery`` – Local workspace scanning (``DiscoveryReport``).
* ``forge_copilot_schema_inference`` – CSV / JSON / Parquet / Avro type inference.

All public names from those modules are **re-exported** here so existing
``from forge_copilot_runtime import …`` statements keep working.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from fluid_build.cli._common import redact_secrets, resolve_provider_from_contract
from fluid_build.cli.forge_copilot_memory import CopilotMemorySnapshot
from fluid_build.cli.forge_copilot_taxonomy import format_use_case_label, normalize_use_case
from fluid_build.schema_manager import FluidSchemaManager
from fluid_build.util.contract import get_builds

# Re-export: LLM providers  ------------------------------------------------
from fluid_build.cli.forge_copilot_llm_providers import (  # noqa: F401  – re-exports
    AnthropicProvider,
    BUILTIN_LLM_PROVIDERS,
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

# Re-export: Discovery  -----------------------------------------------------
from fluid_build.cli.forge_copilot_discovery import (  # noqa: F401  – re-exports
    DiscoveryReport,
    discover_local_context,
)

# Schema inference (only import what this module actually uses)
from fluid_build.cli.forge_copilot_schema_inference import (
    map_inferred_type_to_contract_type as _map_inferred_type_to_contract_type,
)

LOG = logging.getLogger("fluid.cli.forge_copilot")

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


# CopilotGenerationError and LlmConfig are imported from forge_copilot_llm_providers above.


# DiscoveryReport is imported from forge_copilot_discovery above.


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


# LlmProvider and all concrete implementations are imported from
# forge_copilot_llm_providers above.


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


def _normalize_interview_summary(context: Mapping[str, Any]) -> Dict[str, Any]:
    summary = context.get("interview_summary")
    if isinstance(summary, Mapping):
        normalized = dict(summary)
    else:
        normalized = {
            "project_goal": context.get("project_goal"),
            "use_case": normalize_use_case(context.get("use_case")),
            "use_case_other": context.get("use_case_other"),
            "use_case_label": format_use_case_label(
                context.get("use_case"), context.get("use_case_other")
            ),
            "data_sources": context.get("data_sources"),
            "provider_hint": context.get("provider") or context.get("provider_hint"),
            "domain": context.get("domain"),
            "owner_team": context.get("owner_team") or context.get("owner"),
            "build_engine": context.get("build_engine"),
            "output_kind": context.get("output_kind"),
            "semantic_intent": {
                "primary_entity": context.get("primary_entity"),
                "primary_measures": _coerce_string_list(context.get("primary_measures")),
                "primary_dimensions": _coerce_string_list(context.get("primary_dimensions")),
                "time_dimension": context.get("time_dimension"),
                "time_granularity": context.get("time_granularity"),
            },
            "refresh_cadence": context.get("refresh_cadence"),
            "consumes": context.get("consumes") or [],
            "assumptions": list(context.get("assumptions_used") or []),
            "answered_fields": sorted(
                key
                for key in (
                    "project_goal",
                    "use_case",
                    "data_sources",
                    "provider",
                    "domain",
                    "owner_team",
                    "build_engine",
                    "output_kind",
                    "primary_entity",
                    "primary_measures",
                    "primary_dimensions",
                    "time_dimension",
                    "time_granularity",
                    "refresh_cadence",
                    "consumes",
                )
                if context.get(key)
            ),
        }

    semantic_intent = normalized.get("semantic_intent")
    if not isinstance(semantic_intent, Mapping):
        semantic_intent = {}
    normalized["semantic_intent"] = {
        "primary_entity": semantic_intent.get("primary_entity") or context.get("primary_entity"),
        "primary_measures": _coerce_string_list(
            semantic_intent.get("primary_measures") or context.get("primary_measures")
        ),
        "primary_dimensions": _coerce_string_list(
            semantic_intent.get("primary_dimensions") or context.get("primary_dimensions")
        ),
        "time_dimension": semantic_intent.get("time_dimension") or context.get("time_dimension"),
        "time_granularity": semantic_intent.get("time_granularity")
        or context.get("time_granularity"),
    }
    normalized["use_case"] = normalize_use_case(normalized.get("use_case")) or normalized.get("use_case")
    normalized["use_case_label"] = normalized.get("use_case_label") or format_use_case_label(
        normalized.get("use_case"), normalized.get("use_case_other")
    )
    normalized["consumes"] = _normalize_consumes_for_generation(
        normalized.get("consumes") or context.get("consumes")
    )
    normalized["answered_fields"] = sorted(
        set(normalized.get("answered_fields") or [])
        | {
            key
            for key in (
                "project_goal",
                "use_case",
                "data_sources",
                "provider_hint",
                "domain",
                "owner_team",
                "build_engine",
                "output_kind",
                "refresh_cadence",
            )
            if normalized.get(key)
        }
    )
    return normalized


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    text = str(value or "").replace("\n", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _normalize_consumes_for_generation(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        product_id = str(item.get("productId") or item.get("product_id") or "").strip()
        expose_id = str(item.get("exposeId") or item.get("expose_id") or "").strip()
        if product_id and expose_id:
            normalized.append({"productId": product_id, "exposeId": expose_id})
    return normalized


def _build_semantics_from_interview_summary(
    *,
    columns: List[Dict[str, Any]],
    interview_summary: Mapping[str, Any],
    expose_name: str,
    description: str,
) -> Dict[str, Any]:
    semantic_intent = interview_summary.get("semantic_intent")
    if not isinstance(semantic_intent, Mapping):
        semantic_intent = {}

    entity_name = str(semantic_intent.get("primary_entity") or columns[0]["name"]).strip()
    measure_names = _coerce_string_list(semantic_intent.get("primary_measures"))
    if not measure_names:
        measure_names = [f"{entity_name}_count"]
    dimension_names = _coerce_string_list(semantic_intent.get("primary_dimensions"))
    time_dimension = str(semantic_intent.get("time_dimension") or "").strip()
    time_granularity = str(semantic_intent.get("time_granularity") or "").strip()

    entities = [{"name": entity_name, "type": "primary"}]
    measures = []
    metrics = []
    for measure_name in measure_names:
        normalized_measure = sanitize_name(measure_name).replace("-", "_")
        if normalized_measure.endswith("_count"):
            agg = "count"
            expr = entity_name
        else:
            agg = "sum"
            expr = measure_name
        measures.append(
            {
                "name": normalized_measure,
                "agg": agg,
                "expr": expr,
                "description": f"{measure_name} for {expose_name}.",
            }
        )
        metrics.append(
            {
                "name": f"{normalized_measure}_metric",
                "type": "simple",
                "measure": normalized_measure,
                "description": f"Metric for {measure_name}.",
            }
        )

    dimensions = [{"name": entity_name, "type": "categorical"}]
    for dimension_name in dimension_names:
        normalized_dimension = sanitize_name(dimension_name).replace("-", "_")
        if normalized_dimension != entity_name:
            dimensions.append({"name": normalized_dimension, "type": "categorical"})
    if time_dimension:
        time_dimension_entry: Dict[str, Any] = {"name": time_dimension, "type": "time"}
        if time_granularity:
            time_dimension_entry["typeParams"] = {"timeGranularity": time_granularity}
        dimensions.append(time_dimension_entry)

    return {
        "name": expose_name.replace("_", " ").title(),
        "description": description or "Semantic model for the exposed data product.",
        "entities": entities,
        "measures": measures,
        "dimensions": dimensions,
        "metrics": metrics,
    }


def build_seed_contract(
    *,
    context: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    template_name: str,
    provider_name: str,
    project_memory: Optional[CopilotMemorySnapshot] = None,
) -> Dict[str, Any]:
    """Create a minimal valid 0.7.2 contract used as guidance for the LLM."""
    interview_summary = _normalize_interview_summary(context)
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

    build_engine = str(interview_summary.get("build_engine") or "sql").strip().lower()
    if build_engine not in KNOWN_BUILD_ENGINES:
        build_engine = "sql"
    build = {
        "id": "main_build",
        "pattern": "embedded-logic",
        "engine": "python" if build_engine == "python" else "sql",
        "execution": {
            "trigger": {"type": "manual", "iterations": 1},
            "runtime": {
                "platform": provider_name,
                "resources": {"cpu": "1", "memory": "2Gi"},
            },
        },
    }
    if build["engine"] == "python":
        build["repository"] = "src/main.py"
        build["properties"] = {"model": "src.main:build"}
    else:
        build["properties"] = {"sql": "SELECT 1 AS id"}

    description = context.get("project_goal") or "AI-generated FLUID data product"
    domain = (
        context.get("domain")
        or interview_summary.get("domain")
        or (project_memory.preferred_domain if project_memory else None)
        or context.get("use_case")
        or "analytics"
    )
    owner_team = (
        interview_summary.get("owner_team")
        or context.get("owner_team")
        or context.get("owner")
        or (project_memory.preferred_owner if project_memory else None)
        or context.get("team_size")
        or "data-team"
    )
    consumes = interview_summary.get("consumes") or []
    if not isinstance(consumes, list):
        consumes = []

    binding = _default_binding(provider_name, expose_name)
    return {
        "fluidVersion": "0.7.2",
        "kind": "DataProduct",
        "id": f"generated.{project_name}",
        "name": project_name.replace("-", " ").title(),
        "description": description,
        "domain": domain,
        "metadata": {
            "layer": "Bronze",
            "owner": {
                "team": owner_team,
                "email": "data-team@example.com",
            },
        },
        "consumes": consumes,
        "builds": [build],
        "exposes": [
            {
                "exposeId": expose_name,
                "kind": str(interview_summary.get("output_kind") or "table"),
                "binding": binding,
                "contract": {"schema": columns},
                "semantics": _build_semantics_from_interview_summary(
                    columns=columns,
                    interview_summary=interview_summary,
                    expose_name=expose_name,
                    description=description,
                ),
            }
        ],
    }


def build_system_prompt(capability_matrix: Mapping[str, Any]) -> str:
    """System prompt for structured FLUID contract generation."""
    providers = ", ".join(capability_matrix.get("providers") or [])
    engines = ", ".join(capability_matrix.get("build_engines") or sorted(KNOWN_BUILD_ENGINES))
    return (
        "You are FLUID Forge Copilot. Generate a production-ready FLUID 0.7.2 contract and README "
        "that only use locally supported templates, providers, and build engines.\n"
        "Return strict JSON only. Do not wrap the response in markdown fences.\n"
        "Never include secrets, access tokens, raw sample values, or verbatim file contents.\n"
        "ALWAYS use fluidVersion '0.7.2' (Semantic Truth Engine release).\n"
        "Treat project_memory as a soft preference layer only. Explicit user context and the current "
        "discovery report take precedence.\n"
        "Use interview_summary as the authoritative statement of current user intent.\n\n"
        "The JSON object must contain keys: recommended_template, recommended_provider, "
        "recommended_patterns, architecture_suggestions, best_practices, technology_stack, "
        "description, domain, owner, readme_markdown, contract, additional_files.\n\n"
        "CRITICAL: The contract value must be a JSON object that strictly conforms to the FLUID 0.7.2 schema.\n"
        "The ONLY allowed top-level keys in the contract object are: "
        "fluidVersion, kind, id, name, description, domain, metadata, consumes, builds, exposes.\n"
        "DO NOT add 'quality', 'governance', 'owner', or any other top-level key.\n\n"
        "metadata must be an object with: owner (object with team and email) and layer.\n\n"
        "Each build must have: id, pattern (one of: 'embedded-logic', 'hybrid-reference', 'multi-stage'), "
        "engine (one of: " + engines + "), properties, execution.\n"
        "For engine='sql', properties must contain 'sql' with a SQL string.\n"
        "For engine='python', the build must have 'repository' and properties.model.\n"
        "execution must have trigger (object with type and iterations) and runtime (object with platform and resources).\n"
        "DO NOT add 'consumes' or 'produces' inside a build object.\n\n"
        "Each consume must have: productId (string) and exposeId (string). No other keys.\n\n"
        "Each expose must have: exposeId (string), kind (string), binding (object with platform, format, location), "
        "contract (object with schema as array of column objects with name, type, required).\n"
        "binding.platform is REQUIRED and must be one of: " + providers + ".\n"
        "DO NOT put 'platform' inside binding.location.\n\n"
        "NEW IN 0.7.2 — SEMANTICS BLOCK (required on each expose):\n"
        "Each expose MUST include a 'semantics' object with the following structure:\n"
        "- name (string): Human-readable name for this semantic model\n"
        "- description (string): Business context for what this model represents\n"
        "- entities (array): Join keys with type annotations. Each entity has: name (string), "
        "type (one of: 'primary', 'foreign', 'unique', 'natural'), and optional expr and description.\n"
        "- measures (array): Aggregatable expressions. Each measure has: name (string, required), "
        "agg (one of: 'sum', 'avg', 'count', 'count_distinct', 'min', 'max', 'median', 'percentile', required), "
        "and optional expr, description, createMetric (boolean).\n"
        "- dimensions (array): Grouping axes. Each dimension has: name (string, required), "
        "type (one of: 'categorical', 'time', required), and optional expr, description, "
        "typeParams (object with timeGranularity for time dimensions).\n"
        "- metrics (array): KPI definitions. Each metric has: name (string, required), "
        "type (one of: 'simple', 'derived', 'ratio', required), "
        "and optional measure (for simple), filter, inputMetrics (array of strings for derived/ratio), "
        "expr (for derived), numerator/denominator (for ratio), description.\n"
        "The semantics block enables AI agents and BI tools to generate correct queries without hallucination.\n\n"
        "Follow the seed_contract structure exactly as a reference for the correct schema shape.\n"
        f"Allowed providers: {providers}.\n"
        "Only use build engines from the provided capability matrix."
    )


def build_clarification_system_prompt(capability_matrix: Mapping[str, Any]) -> str:
    """System prompt for interview planning before contract generation."""
    providers = ", ".join(capability_matrix.get("providers") or [])
    templates = ", ".join(sorted((capability_matrix.get("templates") or {}).keys()))
    return (
        "You are FLUID Forge Copilot Interview Planner.\n"
        "Your job is to ask the fewest high-signal questions needed to generate a strong FLUID 0.7.2 contract.\n"
        "Return strict JSON only. Do not use markdown fences.\n"
        "Never ask for secrets, passwords, API keys, access tokens, or raw credentials.\n"
        "Use discovery and project memory as context, but explicit current-run user input takes precedence.\n"
        "Ask at most 2 questions in a round. Prefer choices when the taxonomy is stable.\n"
        "Users may answer imperfectly with partial phrases, synonyms, abbreviations, or adjacent concepts.\n"
        "Treat transcript.raw_input as primary evidence of user intent and transcript.resolved_value as a helpful local guess.\n"
        "If local matching is uncertain, prefer inferring from the raw wording over asking a rigid repeat question.\n"
        "Canonical use_case values: analytics, etl_pipeline, streaming, ml_pipeline, data_platform, other.\n"
        "Allowed providers: "
        + providers
        + ". Known templates: "
        + templates
        + ".\n"
        "Return a JSON object with keys: status, reason, context_patch, assumptions, questions.\n"
        "status must be either 'ask' or 'ready'.\n"
        "questions must be an array of objects with: id, field, prompt, type, choices, required, allow_skip.\n"
        "Supported question types are 'text' and 'choice'.\n"
        "Use context_patch to normalize obvious values from existing evidence.\n"
        "Use assumptions only for bounded defaults that are safe to surface to the user.\n"
        "Mark status='ready' when enough intent is known to generate a defensible contract without more questioning."
    )


def build_clarification_user_prompt(
    *,
    interview_state: Mapping[str, Any],
    discovery_report: DiscoveryReport,
    capability_matrix: Mapping[str, Any],
    project_memory: Optional[CopilotMemorySnapshot] = None,
    previous_failure: Sequence[str] | None = None,
) -> str:
    """Build the adaptive interview prompt payload."""
    payload: Dict[str, Any] = {
        "interview_state": interview_state,
        "discovery_report": discovery_report.to_prompt_payload(),
        "capability_matrix": capability_matrix,
        "target_slots": [
            "project_goal",
            "use_case",
            "data_sources",
            "provider_hint",
            "domain",
            "owner_team",
            "build_engine",
            "output_kind",
            "primary_entity",
            "primary_measures",
            "primary_dimensions",
            "time_dimension",
            "time_granularity",
            "refresh_cadence",
            "consumes",
        ],
        "priorities": [
            "Ask nothing if current context and discovery are already sufficient.",
            "Prefer semantic intent questions over generic project-management questions.",
            "If use_case is ambiguous, prefer the canonical taxonomy with an Other / Not sure option.",
            "Assume the user may answer with fuzzy wording and use transcript raw_input plus resolved values together.",
            "If there was a generation failure, only ask questions that directly reduce that ambiguity.",
        ],
    }
    if project_memory:
        payload["project_memory"] = project_memory.to_prompt_payload()
    if previous_failure:
        payload["previous_failure"] = list(previous_failure)
    return json.dumps(payload, indent=2, sort_keys=True)


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
    interview_summary = _normalize_interview_summary(context)
    prompt: Dict[str, Any] = {
        "attempt": attempt_index,
        "interview_summary": interview_summary,
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


_AMBIGUITY_ERROR_KEYWORDS = (
    "semantics",
    "entity",
    "measure",
    "dimension",
    "metric",
    "business context",
    "description",
    "timegranularity",
)
_STRUCTURAL_ERROR_KEYWORDS = (
    "unsupported provider",
    "unsupported template",
    "binding.platform",
    "missing an engine",
    "not supported for provider",
    "must define repository",
    "must include sql",
    "parse error",
)


def classify_generation_failure(attempts: Sequence[GenerationAttemptReport]) -> str:
    """Classify whether a failed run is likely due to missing business intent."""
    combined_errors = " ".join(
        error.lower()
        for attempt in attempts
        for error in ([attempt.parse_error] if attempt.parse_error else []) + attempt.validation_errors
    )
    if not combined_errors:
        return "unknown"
    if any(keyword in combined_errors for keyword in _STRUCTURAL_ERROR_KEYWORDS):
        return "structural"
    if any(keyword in combined_errors for keyword in _AMBIGUITY_ERROR_KEYWORDS):
        return "ambiguous_intent"
    return "structural"


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
        "domain": payload.get("domain")
        or contract.get("domain")
        or _normalize_interview_summary(context).get("domain")
        or context.get("domain")
        or "",
        "owner": payload.get("owner")
        or contract.get("metadata", {}).get("owner", {}).get("team")
        or _normalize_interview_summary(context).get("owner_team"),
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
        semantics = expose.get("semantics")
        if not semantics:
            errors.append(f"Expose '{expose.get('exposeId', 'unknown')}' is missing semantics.")
            continue
        if not (semantics.get("entities") or []):
            errors.append(
                f"Expose '{expose.get('exposeId', 'unknown')}' semantics must include at least one entity."
            )
        if not (semantics.get("measures") or []):
            errors.append(
                f"Expose '{expose.get('exposeId', 'unknown')}' semantics must include at least one measure."
            )
        if not (semantics.get("dimensions") or []):
            errors.append(
                f"Expose '{expose.get('exposeId', 'unknown')}' semantics must include at least one dimension."
            )
        if not (semantics.get("metrics") or []):
            errors.append(
                f"Expose '{expose.get('exposeId', 'unknown')}' semantics must include at least one metric."
            )

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
    """Normalize infrastructure provider aliases (gcp, aws, local, snowflake).

    This is for cloud/infra providers used in contract bindings and scaffold
    decisions.  For LLM provider normalization see
    ``normalize_llm_provider_name`` in ``forge_copilot_llm_providers``.
    """
    if value is None:
        return "local"
    normalized = str(value).strip().lower().replace("-", "_")
    # "claude" / "anthropic" are LLM providers, not infra providers —
    # pass through unchanged so callers don't silently get a wrong mapping.
    return normalized


def sanitize_name(value: Any) -> str:
    """Create a filesystem-safe identifier."""
    text = str(value or "copilot-data-product").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "copilot-data-product"


def redact_secret_like_text(text: str) -> str:
    """Best-effort redaction for common secret patterns."""
    return redact_secrets(text)


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
