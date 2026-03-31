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

"""Prompt builders for the LLM-backed Forge copilot."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from fluid_build.cli.forge_copilot_memory import CopilotMemorySnapshot

from .forge_copilot_contract_helpers import _normalize_interview_summary


def build_system_prompt(
    capability_matrix: Mapping[str, Any], known_build_engines: Sequence[str]
) -> str:
    """System prompt for structured FLUID contract generation."""
    providers = ", ".join(capability_matrix.get("providers") or [])
    engines = ", ".join(capability_matrix.get("build_engines") or list(known_build_engines))
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
        "Allowed providers: " + providers + ". Known templates: " + templates + ".\n"
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
    discovery_report: Any,
    capability_matrix: Mapping[str, Any],
    project_memory: Optional[CopilotMemorySnapshot] = None,
    previous_failure: Sequence[str] | None = None,
) -> str:
    """Build the adaptive interview prompt payload."""
    payload: dict[str, Any] = {
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
    discovery_report: Any,
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
    prompt: dict[str, Any] = {
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
