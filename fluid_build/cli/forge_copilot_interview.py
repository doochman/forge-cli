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

"""Adaptive copilot interview orchestration for interactive forge sessions."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional

from .forge_copilot_runtime import (
    DiscoveryReport,
    LlmConfig,
    build_clarification_system_prompt,
    build_clarification_user_prompt,
    call_llm,
    extract_json_object,
    get_llm_provider,
    normalize_provider_name,
)
from .forge_copilot_taxonomy import (
    USE_CASE_CHOICES,
    format_use_case_label,
    normalize_copilot_context,
    normalize_use_case,
)
from .forge_dialogs import (
    DialogQuestionResult as AskedQuestionResult,
)
from .forge_dialogs import (
    ask_dialog_question as ask_interview_question,
)
from .forge_dialogs import (
    ask_friendly_text,
    normalize_prompt_choices,
    resolve_choice_input,
)

INTERVIEW_MAX_ROUNDS = 3
INTERVIEW_MAX_QUESTIONS_PER_ROUND = 2
INTERVIEW_TRANSCRIPT_WINDOW = 6

SOURCE_PRECEDENCE = {
    "default": 0,
    "clarifier": 1,
    "project_memory": 2,
    "discovery": 3,
    "interactive": 4,
    "explicit": 5,
}

SUMMARY_FIELDS = {
    "project_goal",
    "use_case",
    "use_case_other",
    "data_sources",
    "provider",
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
}

LIST_LIKE_FIELDS = {"primary_measures", "primary_dimensions"}
SCALAR_FIELDS = {
    "project_goal",
    "data_sources",
    "provider",
    "provider_hint",
    "domain",
    "owner_team",
    "build_engine",
    "output_kind",
    "primary_entity",
    "time_dimension",
    "time_granularity",
    "refresh_cadence",
    "use_case_other",
}


@dataclass
class InterviewTurn:
    """Single turn in the copilot interview transcript."""

    role: str
    content: str
    field: str = ""
    question_id: str = ""
    raw_input: str = ""
    resolved_value: Any = None
    resolution_status: str = ""

    def to_payload(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "field": self.field,
            "question_id": self.question_id,
            "raw_input": self.raw_input,
            "resolved_value": self.resolved_value,
            "resolution_status": self.resolution_status,
        }


@dataclass
class InterviewQuestion:
    """A question to present to the user during the adaptive interview."""

    id: str
    field: str
    prompt: str
    type: str = "text"
    choices: List[Dict[str, Any]] = dc_field(default_factory=list)
    required: bool = False
    allow_skip: bool = True
    default: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterviewQuestion":
        field_name = str(
            payload.get("field") or payload.get("key") or payload.get("id") or ""
        ).strip()
        choices = normalize_prompt_choices(list(payload.get("choices") or []))
        if field_name == "use_case" and not choices:
            choices = list(USE_CASE_CHOICES)

        return cls(
            id=str(payload.get("id") or field_name or "question").strip(),
            field=field_name,
            prompt=str(payload.get("prompt") or payload.get("question") or "Tell me more.").strip(),
            type=str(payload.get("type") or "text").strip().lower(),
            choices=choices[: INTERVIEW_MAX_QUESTIONS_PER_ROUND * 4],
            required=bool(payload.get("required", False)),
            allow_skip=bool(payload.get("allow_skip", not payload.get("required", False))),
            default=str(payload.get("default") or "").strip() or None,
        )


@dataclass
class InterviewDecision:
    """LLM-generated decision: ask more questions or proceed to generation."""

    status: str
    reason: str = ""
    context_patch: Dict[str, Any] = dc_field(default_factory=dict)
    assumptions: List[str] = dc_field(default_factory=list)
    questions: List[InterviewQuestion] = dc_field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterviewDecision":
        status = str(payload.get("status") or "ready").strip().lower()
        if status not in {"ask", "ready"}:
            status = "ready"
        context_patch = payload.get("context_patch")
        if not isinstance(context_patch, Mapping):
            context_patch = {}
        assumptions = [
            str(item).strip()
            for item in (payload.get("assumptions") or [])
            if str(item or "").strip()
        ][:6]
        raw_questions = payload.get("questions") or []
        questions = [
            InterviewQuestion.from_payload(raw_question)
            for raw_question in raw_questions[:INTERVIEW_MAX_QUESTIONS_PER_ROUND]
            if isinstance(raw_question, Mapping)
        ]
        return cls(
            status=status,
            reason=str(payload.get("reason") or "").strip(),
            context_patch=dict(context_patch),
            assumptions=assumptions,
            questions=questions,
        )


@dataclass
class CopilotInterviewState:
    """Mutable state accumulated across the adaptive interview rounds."""

    normalized_context: Dict[str, Any] = dc_field(default_factory=dict)
    transcript: List[Dict[str, Any]] = dc_field(default_factory=list)
    answered_fields: set[str] = dc_field(default_factory=set)
    assumptions: List[str] = dc_field(default_factory=list)
    remaining_rounds: int = INTERVIEW_MAX_ROUNDS
    ready: bool = False
    field_sources: Dict[str, str] = dc_field(default_factory=dict)

    def apply_patch(self, patch: Mapping[str, Any], *, source: str) -> None:
        for key, raw_value in patch.items():
            normalized_value = normalize_interview_value(key, raw_value)
            if normalized_value in (None, "", [], {}):
                continue
            current_source = self.field_sources.get(key, "default")
            if SOURCE_PRECEDENCE.get(source, 0) < SOURCE_PRECEDENCE.get(current_source, 0):
                continue
            self.normalized_context[key] = normalized_value
            self.field_sources[key] = source
            if key in SUMMARY_FIELDS:
                self.answered_fields.add(key)
        self.normalized_context = normalize_copilot_context(self.normalized_context)

    def add_assumptions(self, values: List[str]) -> None:
        for value in values:
            if value and value not in self.assumptions:
                self.assumptions.append(value)

    def record_turn(
        self,
        *,
        role: str,
        content: str,
        field: str = "",
        question_id: str = "",
        raw_input: str = "",
        resolved_value: Any = None,
        resolution_status: str = "",
    ) -> None:
        text = str(content or "").strip()
        if not text:
            return
        turn = InterviewTurn(
            role=role,
            content=text,
            field=field,
            question_id=question_id,
            raw_input=str(raw_input or "").strip(),
            resolved_value=resolved_value,
            resolution_status=resolution_status,
        )
        self.transcript.append(turn.to_payload())
        self.transcript = self.transcript[-INTERVIEW_TRANSCRIPT_WINDOW:]

    def to_prompt_payload(self) -> Dict[str, Any]:
        return {
            "normalized_context": dict(self.normalized_context),
            "interview_summary": build_interview_summary_from_context(self.normalized_context),
            "answered_fields": sorted(self.answered_fields),
            "assumptions": list(self.assumptions),
            "remaining_rounds": self.remaining_rounds,
            "transcript": list(self.transcript[-INTERVIEW_TRANSCRIPT_WINDOW:]),
        }

    def finalize(self) -> Dict[str, Any]:
        final_context = normalize_copilot_context(dict(self.normalized_context))
        final_context["interview_summary"] = build_interview_summary_from_context(final_context)
        if self.assumptions:
            final_context["assumptions_used"] = list(self.assumptions)
        return final_context


def normalize_interview_value(field_name: str, value: Any) -> Any:
    """Coerce a raw interview answer into its canonical form for the given field."""
    key = str(field_name or "").strip()
    if key in {"provider", "provider_hint"}:
        text = str(value or "").strip()
        return normalize_provider_name(text) if text else None
    if key == "use_case":
        return normalize_use_case(value) or str(value or "").strip() or None
    if key == "consumes":
        return _normalize_consumes(value)
    if key in LIST_LIKE_FIELDS:
        return _listify_strings(value)
    if key in SCALAR_FIELDS:
        text = str(value or "").strip()
        return text or None
    return value


def _listify_strings(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    text = str(value or "").replace("\n", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _normalize_consumes(value: Any) -> Any:
    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, Mapping):
                product_id = str(item.get("productId") or item.get("product_id") or "").strip()
                expose_id = str(item.get("exposeId") or item.get("expose_id") or "").strip()
                if product_id and expose_id:
                    normalized.append({"productId": product_id, "exposeId": expose_id})
            elif str(item or "").strip():
                normalized.append(str(item).strip())
        return normalized
    if isinstance(value, Mapping):
        product_id = str(value.get("productId") or value.get("product_id") or "").strip()
        expose_id = str(value.get("exposeId") or value.get("expose_id") or "").strip()
        if product_id and expose_id:
            return [{"productId": product_id, "exposeId": expose_id}]
        return []
    text = str(value or "").strip()
    return [text] if text else []


def build_interview_summary_from_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a compact summary dict from the current interview context."""
    existing = context.get("interview_summary")
    if isinstance(existing, Mapping):
        return dict(existing)

    normalized = normalize_copilot_context(dict(context))
    answered_fields = sorted(key for key in SUMMARY_FIELDS if normalized.get(key))
    summary = {
        "project_goal": normalized.get("project_goal"),
        "use_case": normalized.get("use_case"),
        "use_case_other": normalized.get("use_case_other"),
        "use_case_label": format_use_case_label(
            normalized.get("use_case"), normalized.get("use_case_other")
        ),
        "data_sources": normalized.get("data_sources"),
        "provider_hint": normalized.get("provider") or normalized.get("provider_hint"),
        "domain": normalized.get("domain"),
        "owner_team": normalized.get("owner_team") or normalized.get("owner"),
        "build_engine": normalized.get("build_engine"),
        "output_kind": normalized.get("output_kind"),
        "semantic_intent": {
            "primary_entity": normalized.get("primary_entity"),
            "primary_measures": _listify_strings(normalized.get("primary_measures")),
            "primary_dimensions": _listify_strings(normalized.get("primary_dimensions")),
            "time_dimension": normalized.get("time_dimension"),
            "time_granularity": normalized.get("time_granularity"),
        },
        "refresh_cadence": normalized.get("refresh_cadence"),
        "consumes": _normalize_consumes(normalized.get("consumes")),
        "assumptions": list(normalized.get("assumptions_used") or []),
        "answered_fields": answered_fields,
    }
    return summary


def bootstrap_interview_state(
    initial_context: Mapping[str, Any],
    *,
    discovery_report: DiscoveryReport,
    project_memory: Optional[Any] = None,
) -> CopilotInterviewState:
    """Create the initial interview state from explicit context, discovery, and memory."""
    state = CopilotInterviewState()
    state.apply_patch(normalize_copilot_context(dict(initial_context)), source="explicit")

    if not state.normalized_context.get("provider") and discovery_report.provider_hints:
        provider_hint = normalize_provider_name(discovery_report.provider_hints[0])
        if provider_hint:
            state.apply_patch({"provider": provider_hint}, source="discovery")

    if project_memory:
        if not state.normalized_context.get("provider") and getattr(
            project_memory, "preferred_provider", None
        ):
            state.apply_patch(
                {"provider": project_memory.preferred_provider}, source="project_memory"
            )
        if not state.normalized_context.get("domain") and getattr(
            project_memory, "preferred_domain", None
        ):
            state.apply_patch({"domain": project_memory.preferred_domain}, source="project_memory")
        if not state.normalized_context.get("owner_team") and getattr(
            project_memory, "preferred_owner", None
        ):
            state.apply_patch(
                {"owner_team": project_memory.preferred_owner}, source="project_memory"
            )
        memory_engines = list(getattr(project_memory, "build_engines", []) or [])
        if not state.normalized_context.get("build_engine") and memory_engines:
            state.apply_patch({"build_engine": memory_engines[0]}, source="project_memory")

    return state


def run_adaptive_copilot_interview(
    *,
    initial_context: Mapping[str, Any],
    console: Any,
    llm_config: LlmConfig,
    discovery_report: DiscoveryReport,
    capability_matrix: Mapping[str, Any],
    project_memory: Optional[Any] = None,
    previous_failure: Optional[List[str]] = None,
) -> CopilotInterviewState:
    """Run the multi-round adaptive interview, calling the LLM for dynamic questions."""
    state = bootstrap_interview_state(
        initial_context,
        discovery_report=discovery_report,
        project_memory=project_memory,
    )

    _ask_bootstrap_questions(state, console, discovery_report=discovery_report)

    while console and state.remaining_rounds > 0 and not state.ready:
        decision = request_interview_decision(
            state,
            llm_config=llm_config,
            discovery_report=discovery_report,
            capability_matrix=capability_matrix,
            project_memory=project_memory,
            previous_failure=previous_failure,
        )
        if decision.reason:
            state.record_turn(role="assistant", content=decision.reason)
        state.apply_patch(decision.context_patch, source="clarifier")
        state.add_assumptions(decision.assumptions)
        if decision.status == "ready" or not decision.questions:
            state.ready = True
            break
        _ask_dynamic_questions(state, console, decision.questions)
        state.remaining_rounds -= 1
        if not decision.questions:
            break

    state.normalized_context = state.finalize()
    return state


def run_post_generation_clarification(
    state: CopilotInterviewState,
    *,
    console: Any,
    llm_config: LlmConfig,
    discovery_report: DiscoveryReport,
    capability_matrix: Mapping[str, Any],
    project_memory: Optional[Any] = None,
    failure_summary: Optional[List[str]] = None,
) -> CopilotInterviewState:
    """Run one extra clarification round after a generation failure."""
    if not console:
        return state
    decision = request_interview_decision(
        state,
        llm_config=llm_config,
        discovery_report=discovery_report,
        capability_matrix=capability_matrix,
        project_memory=project_memory,
        previous_failure=failure_summary,
    )
    if decision.reason:
        state.record_turn(role="assistant", content=decision.reason)
    state.apply_patch(decision.context_patch, source="clarifier")
    state.add_assumptions(decision.assumptions)
    if decision.status == "ask" and decision.questions:
        _ask_dynamic_questions(state, console, decision.questions)
    state.normalized_context = state.finalize()
    return state


def request_interview_decision(
    state: CopilotInterviewState,
    *,
    llm_config: LlmConfig,
    discovery_report: DiscoveryReport,
    capability_matrix: Mapping[str, Any],
    project_memory: Optional[Any] = None,
    previous_failure: Optional[List[str]] = None,
) -> InterviewDecision:
    """Call the LLM to decide whether to ask more questions or proceed."""
    system_prompt = build_clarification_system_prompt(capability_matrix)
    user_prompt = build_clarification_user_prompt(
        interview_state=state.to_prompt_payload(),
        discovery_report=discovery_report,
        capability_matrix=capability_matrix,
        project_memory=project_memory,
        previous_failure=previous_failure or [],
    )
    raw = call_llm(
        provider=get_llm_provider(llm_config.provider),
        config=llm_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    payload = extract_json_object(raw)
    return InterviewDecision.from_payload(payload)


def _ask_bootstrap_questions(
    state: CopilotInterviewState,
    console: Any,
    *,
    discovery_report: DiscoveryReport,
) -> None:
    if not console:
        return
    if not state.normalized_context.get("project_goal"):
        answer = ask_friendly_text(
            console,
            "What are you trying to build?",
            required=True,
        )
        if answer:
            state.apply_patch({"project_goal": answer}, source="interactive")
            state.record_turn(
                role="user",
                content=answer,
                field="project_goal",
                question_id="bootstrap_project_goal",
                raw_input=answer,
                resolved_value=answer,
                resolution_status="matched",
            )
    if not state.normalized_context.get("data_sources") and _discovery_is_thin(discovery_report):
        answer = ask_friendly_text(
            console,
            "What data sources or systems are involved? (leave blank if you're not sure yet)",
            required=False,
        )
        if answer:
            state.apply_patch({"data_sources": answer}, source="interactive")
            state.record_turn(
                role="user",
                content=answer,
                field="data_sources",
                question_id="bootstrap_data_sources",
                raw_input=answer,
                resolved_value=answer,
                resolution_status="matched",
            )


def _ask_dynamic_questions(
    state: CopilotInterviewState,
    console: Any,
    questions: List[InterviewQuestion],
) -> None:
    for question in questions[:INTERVIEW_MAX_QUESTIONS_PER_ROUND]:
        result = ask_interview_question(console, question)
        if result.context_patch:
            state.apply_patch(result.context_patch, source="interactive")
        content = result.raw_input or str(result.value or "").strip()
        if not content:
            continue
        state.record_turn(
            role="user",
            content=content,
            field=question.field,
            question_id=question.id,
            raw_input=result.raw_input,
            resolved_value=result.value,
            resolution_status=result.resolution_status,
        )


def _discovery_is_thin(discovery_report: DiscoveryReport) -> bool:
    return not any(
        (
            discovery_report.detected_sources,
            discovery_report.sql_files,
            discovery_report.dbt_projects,
            discovery_report.terraform_projects,
            discovery_report.existing_contracts,
            discovery_report.provider_hints,
        )
    )
