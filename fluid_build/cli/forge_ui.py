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

"""Shared Forge user-facing copy and Rich panel helpers."""

from __future__ import annotations
__all__ = [
    "build_copilot_analysis_text",
    "build_standard_next_steps",
    "print_assumptions_panel",
    "print_welcome_panel",
    "show_copilot_analysis",
    "show_domain_analysis",
    "show_lines_panel",
    "show_next_steps_panel",
]


from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    from rich.panel import Panel

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised through non-Rich fallbacks elsewhere
    Panel = None  # type: ignore[assignment]
    RICH_AVAILABLE = False

FORGE_VALIDATE_COMMAND = "fluid validate contract.fluid.yaml"
FORGE_PLAN_COMMAND = "fluid plan contract.fluid.yaml --out runtime/plan.json"
FORGE_APPLY_COMMAND = "fluid apply runtime/plan.json"
FORGE_MARKET_SEARCH_COMMAND = 'fluid market --search "<keyword>"'
FORGE_DIALOG_HINT = (
    "Answer a few questions about your project using a number, short phrase, or your own wording"
)
FORGE_FLEXIBLE_INPUT_SUMMARY = "numbers, short phrases, and natural-language answers"
FORGE_WORKFLOW_STEPS = [
    "Run fluid forge",
    FORGE_DIALOG_HINT,
    "Copilot discovers local metadata and generates a full contract",
    "Forge validates and repairs the contract if needed",
    "Forge scaffolds only after validation passes",
    "Forge shows how memory influenced the run",
    "Save project-scoped memory only if you explicitly opt in",
]


def _build_panel(renderable: str, *, title: str, border_style: str) -> Any:
    if not RICH_AVAILABLE or Panel is None:
        return renderable
    return Panel(renderable, title=title, border_style=border_style)


def show_lines_panel(
    console: Any,
    lines: Sequence[str],
    *,
    title: str,
    border_style: str,
) -> None:
    """Print a simple multi-line panel when Rich is available."""
    if not console or not RICH_AVAILABLE or not lines:
        return
    console.print(_build_panel("\n".join(lines), title=title, border_style=border_style))


def print_welcome_panel(console: Any) -> None:
    """Render the shared Forge welcome panel."""
    if not console or not RICH_AVAILABLE:
        return
    welcome_text = """
🔨 **FLUID Forge** - The one command you need to know

Choose your creation mode:
• **copilot** - AI-powered intelligent project creation (recommended)
• **agent** - Specialized domain experts for specific industries
• **template** - Traditional template-based creation
• **blueprint** - Complete enterprise data product templates
    """.strip()
    console.print(_build_panel(welcome_text, title="Welcome to FLUID Forge", border_style="blue"))


def print_assumptions_panel(console: Any, assumptions: Sequence[str]) -> None:
    """Render the bounded assumptions summary shown after interviews."""
    if not console or not RICH_AVAILABLE or not assumptions:
        return
    assumption_lines = "\n".join(f"• {item}" for item in assumptions[:4])
    console.print(_build_panel(assumption_lines, title="📝 Assumptions Used", border_style="cyan"))


def build_copilot_analysis_text(
    *,
    context: Mapping[str, Any],
    suggestions: Mapping[str, Any],
    use_case_label: str,
    memory_lines: Sequence[str],
) -> str:
    """Build the copilot analysis panel body."""
    lines = [
        f"🎯 **Project Goal:** {context.get('project_goal', 'Not specified')}",
        f"📊 **Data Sources:** {context.get('data_sources', 'Not specified')}",
        f"🏗️ **Use Case:** {use_case_label}",
        f"⚙️ **Complexity:** {context.get('complexity', 'intermediate')}",
        "",
        "🤖 **AI Recommendations:**",
        f"• **Template:** {suggestions['recommended_template']}",
        f"• **Provider:** {suggestions['recommended_provider']}",
        "• **Patterns:** "
        + (", ".join(suggestions.get("recommended_patterns", [])) or "Standard patterns"),
        "",
        "💡 **Architecture Suggestions:**",
    ]

    for suggestion in suggestions.get("architecture_suggestions", []):
        lines.append(f"• {suggestion}")

    best_practices = list(suggestions.get("best_practices", []) or [])
    if best_practices:
        lines.extend(["", "✨ **Best Practices:**"])
        lines.extend(f"• {practice}" for practice in best_practices)

    assumptions = list(context.get("assumptions_used") or [])
    if assumptions:
        lines.extend(["", "📝 **Assumptions Used:**"])
        lines.extend(f"• {item}" for item in assumptions[:4])

    if memory_lines:
        lines.extend(["", "🧠 **Project Memory Guidance:**"])
        lines.extend(f"• {line}" for line in memory_lines)

    return "\n".join(lines)


def show_copilot_analysis(
    console: Any,
    *,
    context: Mapping[str, Any],
    suggestions: Mapping[str, Any],
    use_case_label: str,
    memory_lines: Sequence[str],
) -> None:
    """Render the copilot AI analysis panel."""
    if not console or not RICH_AVAILABLE:
        return
    text = build_copilot_analysis_text(
        context=context,
        suggestions=suggestions,
        use_case_label=use_case_label,
        memory_lines=memory_lines,
    )
    console.print(_build_panel(text.strip(), title="🧠 AI Analysis", border_style="blue"))


def build_domain_analysis_text(
    *,
    goal: str,
    data_sources: str,
    product_type: str,
    suggestions: Mapping[str, Any],
    domain: str,
) -> str:
    """Build the shared domain-agent analysis body."""
    patterns = ", ".join(suggestions.get("recommended_patterns", []) or ["Standard scaffolding"])
    return (
        f"🎯 **Project Goal:** {goal}\n"
        f"📊 **Data Sources:** {data_sources}\n"
        f"🏷️ **Domain Focus:** {product_type}\n\n"
        "🤖 **Recommendations:**\n"
        f"• Template: {suggestions.get('recommended_template')}\n"
        f"• Provider: {suggestions.get('recommended_provider')}\n"
        f"• Patterns: {patterns}\n\n"
        f"[dim]Optimized for {domain} workflows and guardrails.[/dim]"
    )


def show_domain_analysis(
    console: Any,
    *,
    goal: str,
    data_sources: str,
    product_type: str,
    suggestions: Mapping[str, Any],
    domain: str,
) -> None:
    """Render the domain-agent analysis panel."""
    if not console or not RICH_AVAILABLE:
        return
    console.print(
        _build_panel(
            build_domain_analysis_text(
                goal=goal,
                data_sources=data_sources,
                product_type=product_type,
                suggestions=suggestions,
                domain=domain,
            ).strip(),
            title="🧠 AI Analysis",
            border_style="blue",
        )
    )


def build_standard_next_steps(
    *,
    provider: str,
    immediate_validation_text: str,
    provider_setup_text: Optional[str] = None,
    tips_heading: str = "💡 **Pro Tips:**",
    extra_tips: Optional[Iterable[str]] = None,
    footer: Optional[str] = None,
) -> str:
    """Build the shared official-command next-steps text."""
    setup_line = provider_setup_text or f"Configure your {provider} provider credentials"
    lines = [
        "🎯 **Immediate Next Steps:**",
        "1. Review and customize contract.fluid.yaml",
        f"2. Run `{FORGE_VALIDATE_COMMAND}` to {immediate_validation_text}",
        f"3. {setup_line}",
        "",
        "🚀 **Recommended Workflow:**",
        f"1. `{FORGE_VALIDATE_COMMAND}` - Validate your contract",
        f"2. `{FORGE_PLAN_COMMAND}` - Generate execution plan",
        f"3. `{FORGE_APPLY_COMMAND}` - Deploy your data product",
        "",
        tips_heading,
    ]

    tips = list(extra_tips or [])
    lines.extend(f"• {tip}" for tip in tips)
    if footer:
        lines.extend(["", footer])
    return "\n".join(lines)


def show_next_steps_panel(
    console: Any,
    *,
    provider: str,
    immediate_validation_text: str,
    provider_setup_text: Optional[str] = None,
    tips_heading: str = "💡 **Pro Tips:**",
    extra_tips: Optional[Iterable[str]] = None,
    footer: Optional[str] = None,
    title: str = "🚀 What's Next?",
) -> None:
    """Render the shared next-steps panel."""
    if not console or not RICH_AVAILABLE:
        return
    text = build_standard_next_steps(
        provider=provider,
        immediate_validation_text=immediate_validation_text,
        provider_setup_text=provider_setup_text,
        tips_heading=tips_heading,
        extra_tips=extra_tips,
        footer=footer,
    )
    console.print(_build_panel(text.strip(), title=title, border_style="green"))


def build_blueprint_next_steps() -> str:
    """Build the shared blueprint next-steps panel body."""
    lines = [
        "🎯 **Immediate Next Steps:**",
        "1. Review and customize contract.fluid.yaml",
        "2. Run `fluid validate contract.fluid.yaml` to check the blueprint contract",
        "3. Update provider-specific configuration before planning",
        "",
        "🚀 **Recommended Workflow:**",
        f"1. `{FORGE_VALIDATE_COMMAND}` - Validate your contract",
        f"2. `{FORGE_PLAN_COMMAND}` - Generate execution plan",
        f"3. `{FORGE_APPLY_COMMAND}` - Deploy your data product",
        "",
        "💡 **Blueprint Tips:**",
        "• Review scaffolded environment variables and provider settings first",
        "• Run `fluid doctor` if your provider setup needs a quick sanity check",
        "",
        "[dim]Generated by FLUID Forge Blueprint Mode[/dim]",
    ]
    return "\n".join(lines)


def show_blueprint_next_steps(console: Any) -> None:
    """Render the blueprint next-steps panel."""
    if not console or not RICH_AVAILABLE:
        return
    console.print(
        _build_panel(
            build_blueprint_next_steps().strip(), title="🚀 What's Next?", border_style="green"
        )
    )
