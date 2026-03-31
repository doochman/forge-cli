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

"""
FLUID Build - Forge Command (Enhanced with AI Agents)

The one command you need to know for creating FLUID data products.
Supports multiple creation modes:
- Templates: Traditional template-based project creation
- AI Copilot: AI-powered intelligent project creation
- Domain Agents: Specialized AI agents for specific domains
- Blueprints: Complete data product templates

This unified interface maintains the simplicity of "fluid forge" while
providing powerful AI-assisted development capabilities.
"""

import argparse
import json
import logging
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fluid_build.cli.console import cprint, success, warning
from fluid_build.cli.console import error as console_error

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.status import Status
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..blueprints import registry as blueprint_registry
from ._common import CLIError
from .forge_copilot_memory import (
    CopilotMemoryStore,
    build_copilot_project_memory,
    resolve_copilot_memory_root,
    summarize_copilot_memory,
)
from .forge_copilot_runtime import (
    CopilotGenerationError,
    CopilotGenerationResult,
    build_capability_matrix,
    discover_local_context,
    generate_copilot_artifacts,
    normalize_provider_name,
    normalize_template_name,
    resolve_llm_config,
)

COMMAND = "forge"
LOG = logging.getLogger("fluid.cli.forge")


# Custom Exceptions for better error handling
class ForgeError(CLIError):
    """Base exception for Forge command errors"""

    pass


class TemplateNotFoundError(ForgeError):
    """Template not found in registry"""

    def __init__(self, template_name: str, available: List[str]):
        self.template_name = template_name
        self.available = available
        super().__init__(f"Template '{template_name}' not found. Available: {', '.join(available)}")


class BlueprintNotFoundError(ForgeError):
    """Blueprint not found in registry"""

    def __init__(self, blueprint_name: str, available: List[str]):
        self.blueprint_name = blueprint_name
        self.available = available
        super().__init__(
            f"Blueprint '{blueprint_name}' not found. Available: {', '.join(available)}"
        )


class InvalidProjectNameError(ForgeError):
    """Invalid project name format"""

    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason
        super().__init__(f"Invalid project name '{name}': {reason}")


class ProjectGenerationError(ForgeError):
    """Project generation failed"""

    pass


class ContextValidationError(ForgeError):
    """Context file validation failed"""

    pass


class ForgeMode(Enum):
    """Forge creation modes"""

    TEMPLATE = "template"
    AI_COPILOT = "copilot"
    DOMAIN_AGENT = "agent"
    BLUEPRINT = "blueprint"


class AIAgent:
    """Base class for AI agents"""

    def __init__(self, name: str, description: str, domain: str):
        self.name = name
        self.description = description
        self.domain = domain
        self.console = Console() if RICH_AVAILABLE else None

    async def create_project(self, target_dir: Path, context: Dict[str, Any]) -> bool:
        """Create project using AI agent"""
        raise NotImplementedError

    def get_questions(self) -> List[Dict[str, Any]]:
        """Get questions for this agent"""
        raise NotImplementedError


class CopilotAgent(AIAgent):
    """General-purpose AI Copilot agent"""

    def __init__(self):
        super().__init__(
            name="copilot",
            description="General-purpose AI assistant for data product creation",
            domain="general",
        )
        self._project_memory_enabled = True
        self._project_memory_snapshot = None
        self._project_memory_path = None

    def get_questions(self) -> List[Dict[str, Any]]:
        """Get copilot questions"""
        return [
            {
                "key": "project_goal",
                "question": "What are you trying to build? (e.g., 'Customer analytics dashboard', 'ML recommendation engine')",
                "type": "text",
                "required": True,
            },
            {
                "key": "data_sources",
                "question": "What data sources will you use? (e.g., 'BigQuery tables', 'REST APIs', 'CSV files')",
                "type": "text",
                "required": True,
            },
            {
                "key": "use_case",
                "question": "What's your primary use case?",
                "type": "choice",
                "choices": ["analytics", "ml_pipeline", "data_lake", "real_time", "reporting"],
                "required": True,
            },
            {
                "key": "team_size",
                "question": "How large is your team?",
                "type": "choice",
                "choices": ["solo", "small (2-5)", "medium (6-15)", "large (15+)"],
                "required": False,
            },
            {
                "key": "complexity",
                "question": "Preferred complexity level?",
                "type": "choice",
                "choices": ["simple", "intermediate", "advanced"],
                "default": "intermediate",
            },
        ]

    def analyze_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze requirements and provide intelligent suggestions"""
        use_case = context.get("use_case", "analytics").lower()
        complexity = context.get("complexity", "intermediate").lower()
        data_sources = context.get("data_sources", "").lower()

        suggestions = {
            "recommended_template": "starter",  # Default fallback
            "recommended_provider": "local",  # Safe default
            "recommended_patterns": [],
            "architecture_suggestions": [],
            "best_practices": [],
            "technology_stack": [],
        }

        # Template selection based on use case
        if "machine learning" in use_case or "ml" in use_case or "ml_pipeline" in use_case:
            suggestions["recommended_template"] = "ml_pipeline"
        elif "streaming" in use_case or "real_time" in use_case:
            suggestions["recommended_template"] = "streaming"
        elif "etl" in use_case or "pipeline" in use_case:
            suggestions["recommended_template"] = "etl_pipeline"
        elif "analytics" in use_case or "reporting" in use_case:
            suggestions["recommended_template"] = "analytics"
        else:
            suggestions["recommended_template"] = "starter"  # Safe fallback

        # Provider selection based on data sources
        if "bigquery" in data_sources or "gcp" in data_sources:
            suggestions["recommended_provider"] = "gcp"
        elif "snowflake" in data_sources:
            suggestions["recommended_provider"] = "snowflake"
        elif "aws" in data_sources:
            suggestions["recommended_provider"] = "aws"
        else:
            suggestions["recommended_provider"] = "local"  # Safe for development

        # Pattern recommendations
        if complexity == "advanced":
            suggestions["recommended_patterns"] = ["data_mesh", "microservices"]
        elif complexity == "intermediate":
            suggestions["recommended_patterns"] = ["layered_architecture"]
        else:
            suggestions["recommended_patterns"] = ["simple_pipeline"]

        # Architecture suggestions
        suggestions["architecture_suggestions"].extend(
            [
                "Start with modular design",
                "Implement proper data validation",
                "Set up monitoring and logging",
            ]
        )

        # Best practices
        suggestions["best_practices"].extend(
            [
                "Use version control for all code",
                "Implement automated testing",
                "Document data lineage",
                "Set up CI/CD pipeline",
            ]
        )

        return suggestions

    def generate_project_artifacts(
        self, context: Dict[str, Any], copilot_options: Optional[Dict[str, Any]] = None
    ) -> CopilotGenerationResult:
        """Generate and validate copilot artifacts via the LLM runtime."""
        options = SimpleNamespace(**(copilot_options or {}))
        llm_config = resolve_llm_config(options)
        target_dir = getattr(options, "target_dir", None)
        target_path = Path(target_dir).expanduser() if target_dir else None
        discovery_report = discover_local_context(
            getattr(options, "discovery_path", None),
            discover=getattr(options, "discover", True),
            workspace_root=Path.cwd(),
            logger=LOG,
        )
        project_memory = self._load_project_memory(
            enabled=getattr(options, "memory", True),
            target_dir=target_path,
        )
        return generate_copilot_artifacts(
            context,
            llm_config=llm_config,
            discovery_report=discovery_report,
            project_memory=project_memory,
            capability_matrix=build_capability_matrix(),
            logger=LOG,
        )

    def create_project(
        self,
        target_dir: Path,
        context: Dict[str, Any],
        copilot_options: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> bool:
        """Create a project from validated LLM-generated artifacts."""
        try:
            options = dict(copilot_options or {})
            options.setdefault("target_dir", str(target_dir))
            generation_result = self.generate_project_artifacts(context, options)
            suggestions = generation_result.suggestions

            # Show AI analysis to user
            self._show_ai_analysis(context, suggestions, generation_result)

            # Create project configuration for ForgeEngine
            project_config = self._create_forge_config(
                target_dir, context, suggestions, generation_result
            )

            # Use ForgeEngine to create and validate the project properly
            success = self._create_with_forge_engine(project_config, dry_run=dry_run)

            if success:
                self._maybe_save_project_memory(
                    target_dir=target_dir,
                    context=context,
                    suggestions=suggestions,
                    generation_result=generation_result,
                    copilot_options=options,
                    dry_run=dry_run,
                )
                # Show next steps
                self._show_next_steps(target_dir, context, suggestions)
                return True
            else:
                if self.console:
                    self.console.print("[red]❌ Project creation failed validation[/red]")
                return False

        except CopilotGenerationError as e:
            if self.console:
                self.console.print(f"[red]❌ {e.message}[/red]")
                for suggestion in e.suggestions:
                    self.console.print(f"[dim]• {suggestion}[/dim]")
            else:
                console_error(e.message)
                for suggestion in e.suggestions:
                    cprint(f"  • {suggestion}")
            return False
        except Exception as e:
            if self.console:
                self.console.print(f"[red]❌ Failed to create project: {e}[/red]")
            else:
                console_error(f"Failed to create project: {e}")
            return False

    def _create_forge_config(
        self,
        target_dir: Path,
        context: Dict[str, Any],
        suggestions: Dict[str, Any],
        generation_result: Optional[CopilotGenerationResult] = None,
    ) -> Dict[str, Any]:
        """Create configuration for ForgeEngine"""
        goal = context.get("project_goal", "Data Product")
        contract_name = (
            generation_result.contract.get("name")
            if generation_result and generation_result.contract
            else goal
        )

        # Create project name that will pass validation
        project_name = self._sanitize_project_name(contract_name)

        config = {
            "name": project_name,
            "description": suggestions.get("description") or f"AI-generated {goal}",
            "domain": suggestions.get("domain") or context.get("domain") or "analytics",
            "owner": suggestions.get("owner") or "data-team",
            "template": normalize_template_name(suggestions["recommended_template"]),
            "provider": normalize_provider_name(suggestions["recommended_provider"]),
            "target_dir": str(target_dir),
            "ai_context": context,
            "ai_suggestions": suggestions,
        }

        if generation_result:
            config["fluid_version"] = generation_result.contract.get("fluidVersion", "0.7.1")
            config["copilot_generated_contract"] = generation_result.contract
            config["copilot_generated_readme"] = generation_result.readme_markdown
            if generation_result.additional_files:
                config["copilot_generated_files"] = generation_result.additional_files

        return config

    def _load_project_memory(self, *, enabled: bool, target_dir: Optional[Path]) -> Optional[Any]:
        """Load repo-local copilot memory when enabled for the current run."""
        project_root = resolve_copilot_memory_root(Path.cwd(), target_dir=target_dir)
        store = CopilotMemoryStore(project_root, logger=LOG)
        self._project_memory_enabled = enabled
        self._project_memory_path = store.path
        self._project_memory_snapshot = None

        if not enabled:
            self._emit_memory_load_feedback()
            return None

        memory = store.load()
        if not memory:
            self._emit_memory_load_feedback()
            return None
        self._project_memory_snapshot = memory.to_prompt_snapshot()
        self._emit_memory_load_feedback()
        return self._project_memory_snapshot

    def _emit_memory_load_feedback(self) -> None:
        """Tell the user how project memory will be handled for this run."""
        if self.console:
            summary_lines = self._build_memory_status_lines()
            self.console.print(
                Panel("\n".join(summary_lines), title="🧠 Project Memory", border_style="cyan")
            )
            return

        for line in self._build_memory_status_lines():
            cprint(line)

    def _build_memory_status_lines(self) -> List[str]:
        """Build short human-facing status lines for current memory state."""
        relative_path = self._relative_memory_path()
        if not self._project_memory_enabled:
            return [f"Project memory is disabled for this run (`--no-memory`). Path: `{relative_path}`"]
        if not self._project_memory_snapshot:
            return [
                "No project-scoped copilot memory was found yet.",
                f"Copilot will rely on your current answers and discovery only. Path: `{relative_path}`",
            ]

        summary = summarize_copilot_memory(self._project_memory_snapshot)
        lines = [f"Loaded project memory from `{relative_path}`."]
        profile = ", ".join(
            [
                part
                for part in (
                    f"template={summary.get('preferred_template')}" if summary.get("preferred_template") else "",
                    f"provider={summary.get('preferred_provider')}" if summary.get("preferred_provider") else "",
                    f"domain={summary.get('preferred_domain')}" if summary.get("preferred_domain") else "",
                )
                if part
            ]
        )
        if profile:
            lines.append(f"Saved profile: {profile}")
        if summary.get("build_engines"):
            lines.append(f"Remembered build engines: {', '.join(summary['build_engines'])}")
        lines.append(
            "Saved schema summaries: "
            f"{summary.get('schema_summary_count', 0)}; recent successful outcomes: {summary.get('recent_outcome_count', 0)}"
        )
        return lines

    def _relative_memory_path(self) -> str:
        """Render the memory path relative to the current directory when possible."""
        if not self._project_memory_path:
            return "runtime/.state/copilot-memory.json"
        try:
            return self._project_memory_path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return str(self._project_memory_path)

    def _build_memory_save_preview_lines(self, memory) -> List[str]:
        """Summarize what will be persisted if the user opts in to saving memory."""
        summary = summarize_copilot_memory(memory)
        lines = [f"Forge will save project memory to `{self._relative_memory_path()}` with:"]
        profile = ", ".join(
            [
                part
                for part in (
                    f"template={summary.get('preferred_template')}" if summary.get("preferred_template") else "",
                    f"provider={summary.get('preferred_provider')}" if summary.get("preferred_provider") else "",
                    f"domain={summary.get('preferred_domain')}" if summary.get("preferred_domain") else "",
                    f"owner={summary.get('preferred_owner')}" if summary.get("preferred_owner") else "",
                )
                if part
            ]
        )
        if profile:
            lines.append(profile)
        if summary.get("build_engines"):
            lines.append(f"build_engines={', '.join(summary['build_engines'])}")
        if summary.get("source_formats"):
            source_bits = ", ".join(
                f"{key}={value}" for key, value in sorted(summary["source_formats"].items())
            )
            lines.append(f"source_formats={source_bits}")
        lines.append(
            "bounded summaries: "
            f"{summary.get('schema_summary_count', 0)} schema summaries, {summary.get('recent_outcome_count', 0)} recent outcomes"
        )
        return lines

    def _maybe_save_project_memory(
        self,
        *,
        target_dir: Path,
        context: Dict[str, Any],
        suggestions: Dict[str, Any],
        generation_result: CopilotGenerationResult,
        copilot_options: Dict[str, Any],
        dry_run: bool,
    ) -> None:
        """Persist project-scoped memory only after a successful non-dry-run scaffold."""
        if dry_run:
            return

        options = SimpleNamespace(**(copilot_options or {}))
        store = CopilotMemoryStore(target_dir, logger=LOG)
        candidate_memory = build_copilot_project_memory(
            project_root=target_dir,
            context=context,
            suggestions=suggestions,
            contract=generation_result.contract,
            discovery_report=generation_result.discovery_report,
            existing_memory=store.load(),
        )

        should_save = self._should_save_project_memory(options, candidate_memory)
        if not should_save:
            if getattr(options, "non_interactive", False) and not getattr(options, "save_memory", False):
                note = "Project memory was not saved. Re-run with `--save-memory` to remember these conventions."
                if self.console:
                    self.console.print(f"[dim]{note}[/dim]")
                else:
                    cprint(note)
            return

        try:
            store.save(candidate_memory)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Failed to save copilot memory at %s: %s", store.path, exc)
            if self.console:
                self.console.print(
                    f"[yellow]⚠ Could not save copilot memory to runtime/.state/copilot-memory.json: {exc}[/yellow]"
                )
            else:
                warning(f"Could not save copilot memory: {exc}")
            return

        if self.console:
            self.console.print(
                "[green]✓[/green] Saved project-scoped copilot memory to runtime/.state/copilot-memory.json"
            )
        else:
            success("Saved project-scoped copilot memory to runtime/.state/copilot-memory.json")

    def _should_save_project_memory(self, options: SimpleNamespace, memory) -> bool:
        """Determine whether this successful run should persist project memory."""
        if getattr(options, "non_interactive", False):
            return bool(getattr(options, "save_memory", False))

        prompt = "Save project-scoped copilot memory to runtime/.state/copilot-memory.json?"
        try:
            if self.console and RICH_AVAILABLE:
                preview = "\n".join(self._build_memory_save_preview_lines(memory))
                self.console.print(
                    Panel(preview, title="🧠 Save Project Memory?", border_style="cyan")
                )
                return Confirm.ask(prompt, default=False, console=self.console)
            for line in self._build_memory_save_preview_lines(memory):
                cprint(line)
            answer = input(f"{prompt} [y/N]: ").strip().lower()
        except Exception:  # noqa: BLE001
            return False
        return answer in {"y", "yes"}

    def _sanitize_project_name(self, goal: str) -> str:
        """Create a valid project name from goal"""
        from .forge_validation import sanitize_project_name

        return sanitize_project_name(goal, strict=False)

    def _create_with_forge_engine(self, project_config: Dict[str, Any], dry_run: bool = False) -> bool:
        """Use ForgeEngine to create and validate project"""
        try:
            from ..forge import ForgeEngine

            if self.console and RICH_AVAILABLE:
                with self.console.status(
                    "[bold blue]🔧 Generating project...", spinner="dots"
                ) as status:
                    status.update(
                        f"[dim]Using template: {project_config.get('template', 'N/A')}[/dim]"
                    )

                    # Create ForgeEngine instance
                    engine = ForgeEngine()

                    status.update("[dim]Validating configuration...[/dim]")
                    # Use the engine's run_with_config method for full validation and generation
                    success = engine.run_with_config(project_config, dry_run=dry_run)

                    if success:
                        status.update("[green]✓ Project generated successfully[/green]")

                return success
            else:
                # No Rich available, simple mode
                cprint("🔧 Generating project...")
                engine = ForgeEngine()
                success = engine.run_with_config(project_config, dry_run=dry_run)
                if success:
                    cprint("✓ Project generated successfully")
                return success

        except Exception as e:
            if self.console:
                self.console.print(f"[red]❌ ForgeEngine integration failed: {e}[/red]")
            else:
                console_error(f"ForgeEngine integration failed: {e}")
            return False

    def _analyze_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user requirements and generate suggestions"""
        goal = context.get("project_goal", "").lower()
        data_sources = context.get("data_sources", "").lower()
        context.get("use_case", "analytics")
        complexity = context.get("complexity", "intermediate")

        suggestions = {
            "recommended_template": "analytics",
            "recommended_provider": "local",
            "recommended_patterns": [],
            "architecture_suggestions": [],
            "best_practices": [],
        }

        # Analyze goal keywords
        if any(word in goal for word in ["ml", "machine learning", "model", "prediction"]):
            suggestions["recommended_template"] = "ml_pipeline"
            suggestions["recommended_patterns"].append("feature_store")
            suggestions["architecture_suggestions"].append("Consider MLflow for model versioning")
        elif any(word in goal for word in ["dashboard", "reporting", "analytics", "visualization"]):
            suggestions["recommended_template"] = "analytics"
            suggestions["recommended_patterns"].append("dimensional_modeling")
            suggestions["architecture_suggestions"].append(
                "Use layered architecture (bronze/silver/gold)"
            )
        elif any(word in goal for word in ["real-time", "streaming", "live"]):
            suggestions["recommended_template"] = "streaming"
            suggestions["recommended_patterns"].append("event_sourcing")
            suggestions["architecture_suggestions"].append("Consider Apache Kafka for streaming")

        # Analyze data sources
        if "bigquery" in data_sources:
            suggestions["recommended_provider"] = "gcp"
            suggestions["best_practices"].append("Use BigQuery partitioning for large datasets")
        elif "snowflake" in data_sources:
            suggestions["recommended_provider"] = "snowflake"
            suggestions["best_practices"].append("Leverage Snowflake's auto-scaling features")
        elif any(word in data_sources for word in ["s3", "redshift", "athena"]):
            suggestions["recommended_provider"] = "aws"
            suggestions["best_practices"].append("Use S3 for cost-effective data lake storage")

        # Complexity-based suggestions
        if complexity == "simple":
            suggestions["architecture_suggestions"].append("Start with single-layer architecture")
            suggestions["best_practices"].append("Focus on essential features first")
        elif complexity == "advanced":
            suggestions["recommended_patterns"].extend(["data_mesh", "event_driven"])
            suggestions["architecture_suggestions"].append("Consider microservices architecture")

        return suggestions

    def _show_ai_analysis(
        self,
        context: Dict[str, Any],
        suggestions: Dict[str, Any],
        generation_result: Optional[CopilotGenerationResult] = None,
    ):
        """Show AI analysis and recommendations"""
        if not self.console:
            return

        analysis_text = f"""
🎯 **Project Goal:** {context.get('project_goal', 'Not specified')}
📊 **Data Sources:** {context.get('data_sources', 'Not specified')}
🏗️ **Use Case:** {context.get('use_case', 'analytics')}
⚙️ **Complexity:** {context.get('complexity', 'intermediate')}

🤖 **AI Recommendations:**
• **Template:** {suggestions['recommended_template']}
• **Provider:** {suggestions['recommended_provider']}
• **Patterns:** {', '.join(suggestions['recommended_patterns']) or 'Standard patterns'}

💡 **Architecture Suggestions:**
"""

        for suggestion in suggestions["architecture_suggestions"]:
            analysis_text += f"• {suggestion}\n"

        if suggestions["best_practices"]:
            analysis_text += "\n✨ **Best Practices:**\n"
            for practice in suggestions["best_practices"]:
                analysis_text += f"• {practice}\n"

        memory_lines = self._build_memory_guidance_lines(generation_result)
        if memory_lines:
            analysis_text += "\n🧠 **Project Memory Guidance:**\n"
            for line in memory_lines:
                analysis_text += f"• {line}\n"

        self.console.print(
            Panel(analysis_text.strip(), title="🧠 AI Analysis", border_style="blue")
        )

    def _build_memory_guidance_lines(
        self,
        generation_result: Optional[CopilotGenerationResult],
    ) -> List[str]:
        """Explain how project memory did or did not influence this run."""
        lines: List[str] = []
        if not self._project_memory_enabled:
            return ["Disabled for this run with `--no-memory`."]
        if not self._project_memory_snapshot:
            return ["No saved project memory was available, so only current context and discovery were used."]
        summary = summarize_copilot_memory(self._project_memory_snapshot)
        lines.append(
            "Loaded saved conventions"
            + (
                f" (`{summary.get('preferred_template')}` / `{summary.get('preferred_provider')}`)"
                if summary.get("preferred_template") or summary.get("preferred_provider")
                else ""
            )
            + "."
        )
        decision = generation_result.scaffold_decision if generation_result else None
        if not decision:
            return lines

        lines.append(
            f"Template seed: `{decision.template}` from {self._friendly_source_name(decision.template_source)}."
        )
        lines.append(
            f"Provider seed: `{decision.provider}` from {self._friendly_source_name(decision.provider_source)}."
        )
        if decision.template_source != "project_memory" or decision.provider_source != "project_memory":
            lines.append("Saved memory was treated as a soft preference and did not override stronger current signals.")
        return lines

    def _friendly_source_name(self, source: Optional[str]) -> str:
        """Render scaffold-decision sources in user-friendly language."""
        mapping = {
            "explicit_context": "your explicit input",
            "current_discovery": "current discovery",
            "heuristic_context": "your current answers",
            "project_memory": "saved project memory",
            "default": "safe defaults",
        }
        return mapping.get(source or "", "seed guidance")

    async def _generate_intelligent_structure(
        self, target_dir: Path, context: Dict[str, Any], suggestions: Dict[str, Any]
    ):
        """Generate intelligent project structure"""
        if not self.console:
            return

        with self.console.status("[bold blue]🤖 AI is crafting your project..."):
            # Create directory structure
            target_dir.mkdir(parents=True, exist_ok=True)

            # Generate contract based on AI analysis
            contract = self._generate_intelligent_contract(context, suggestions)

            # Write contract file
            contract_path = target_dir / "contract.fluid.yaml"
            with open(contract_path, "w") as f:
                f.write(contract)

            # Generate supporting files
            self._generate_supporting_files(target_dir, context, suggestions)

            # Generate README with AI insights
            readme = self._generate_intelligent_readme(context, suggestions)
            with open(target_dir / "README.md", "w") as f:
                f.write(readme)

        self.console.print(f"[green]✅ AI-generated project created at {target_dir}[/green]")

    def _generate_intelligent_contract(
        self, context: Dict[str, Any], suggestions: Dict[str, Any]
    ) -> str:
        """Generate intelligent contract based on AI analysis"""
        goal = context.get("project_goal", "Data Product")
        use_case = context.get("use_case", "analytics")
        provider = suggestions["recommended_provider"]

        # Basic contract structure
        contract = f"""# FLUID Contract - AI Generated
# Goal: {goal}
# Generated by AI Copilot on {Path.cwd()}

meta:
  name: {goal.lower().replace(' ', '-')}
  version: "1.0.0"
  description: "{goal}"
  owner: data-team
  domain: {use_case}
  
sources:
  - name: raw_data
    type: table
    description: "Primary data source for {goal}"
    location: "{{{{ provider.dataset }}}}.raw_data"
    
transforms:
  - name: clean_data
    type: sql
    description: "Data cleaning and validation"
    materialization: table
    sources:
      - ref("raw_data")
    sql: |
      SELECT 
        *,
        CURRENT_TIMESTAMP() as processed_at
      FROM {{{{ ref("raw_data") }}}}
      WHERE data_quality_score > 0.8
      
  - name: aggregated_metrics
    type: sql
    description: "Business metrics aggregation"
    materialization: table
    sources:
      - ref("clean_data")
    sql: |
      SELECT
        date_trunc(created_date, DAY) as metric_date,
        count(*) as total_records,
        avg(value) as avg_value
      FROM {{{{ ref("clean_data") }}}}
      GROUP BY 1
      ORDER BY 1 DESC

exposures:
  - name: {goal.lower().replace(' ', '_')}_dataset
    type: table
    description: "Final dataset for {goal}"
    sources:
      - ref("aggregated_metrics")
    
provider:
  type: {provider}
  {"# GCP-specific configuration" if provider == "gcp" else ""}
  {"project: your-gcp-project" if provider == "gcp" else ""}
  {"dataset: your_dataset" if provider == "gcp" else ""}
  
quality:
  - name: data_freshness
    description: "Ensure data is updated daily"
    test: "max_age(1, 'day')"
    
  - name: completeness_check
    description: "Check for required fields"
    test: "not_null(['id', 'created_date'])"
"""

        return contract

    def _generate_supporting_files(
        self, target_dir: Path, context: Dict[str, Any], suggestions: Dict[str, Any]
    ):
        """Generate supporting files"""
        # Generate .gitignore
        gitignore = """
# FLUID Build artifacts
runtime/
.fluid/
*.log

# Provider-specific
.env
credentials.json

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
"""
        with open(target_dir / ".gitignore", "w") as f:
            f.write(gitignore.strip())

        # Generate Makefile for common tasks
        makefile = """
# FLUID Project Makefile
# Generated by AI Copilot

.PHONY: validate plan apply clean help

help:
\t@echo "Available commands:"
\t@echo "  validate  - Validate the contract"
\t@echo "  plan      - Generate execution plan"
\t@echo "  apply     - Apply the plan"
\t@echo "  clean     - Clean up artifacts"

validate:
\tfluid validate contract.fluid.yaml

plan:
\tfluid plan contract.fluid.yaml --out runtime/plan.json

apply:
\tfluid apply runtime/plan.json

clean:
\trm -rf runtime/
\trm -f *.log
"""
        with open(target_dir / "Makefile", "w") as f:
            f.write(makefile)

    def _generate_intelligent_readme(
        self, context: Dict[str, Any], suggestions: Dict[str, Any]
    ) -> str:
        """Generate intelligent README with AI insights"""
        goal = context.get("project_goal", "Data Product")
        use_case = context.get("use_case", "analytics")

        readme = f"""# {goal}

> AI-Generated FLUID Data Product

## Overview

This project was intelligently generated by FLUID AI Copilot based on your requirements:

- **Goal:** {goal}
- **Use Case:** {use_case}
- **Data Sources:** {context.get('data_sources', 'Various sources')}
- **Recommended Template:** {suggestions['recommended_template']}

## AI Insights

### Architecture Recommendations
"""

        for suggestion in suggestions["architecture_suggestions"]:
            readme += f"- {suggestion}\n"

        readme += "\n### Best Practices\n"
        for practice in suggestions["best_practices"]:
            readme += f"- {practice}\n"

        readme += f"""
## Quick Start

1. **Validate your contract:**
   ```bash
   fluid validate contract.fluid.yaml
   ```

2. **Generate execution plan:**
   ```bash
   fluid plan contract.fluid.yaml --out runtime/plan.json
   ```

3. **Apply the plan:**
   ```bash
   fluid apply runtime/plan.json
   ```

## Project Structure

```
{goal.lower().replace(' ', '-')}/
├── contract.fluid.yaml    # Main FLUID contract
├── README.md             # This file
├── Makefile             # Common tasks
└── .gitignore           # Git ignore rules
```

## Next Steps

1. Customize the contract based on your specific data sources
2. Add provider-specific configurations
3. Define data quality rules
4. Set up CI/CD pipeline with `fluid scaffold-ci`

## AI Copilot Generated

This project structure was intelligently created by FLUID AI Copilot.
For more advanced features, explore:

- `fluid forge --mode agent --agent domain-expert` for domain-specific assistance
- `fluid forge --mode blueprint` for complete enterprise templates
- `fluid market` for discovering existing data products

---

*Generated by FLUID AI Copilot - The one command you need to know: `fluid forge`*
"""

        return readme

    def _show_next_steps(
        self, target_dir: Path, context: Dict[str, Any], suggestions: Dict[str, Any]
    ):
        """Show intelligent next steps"""
        if not self.console:
            return

        next_steps = f"""
🎯 **Immediate Next Steps:**
1. Review and customize contract.fluid.yaml
2. Run `fluid validate contract.fluid.yaml` to check your setup
3. Configure your {suggestions['recommended_provider']} provider credentials

🚀 **Recommended Workflow:**
1. `make validate` - Validate your contract
2. `make plan` - Generate execution plan  
3. `make apply` - Deploy your data product

💡 **Pro Tips:**
• Use `fluid market search` to discover similar data products
• Run `fluid doctor` if you encounter any issues
• Check `fluid auth status` for provider authentication
• Use `fluid forge --show-memory` to inspect saved copilot conventions
        """

        self.console.print(Panel(next_steps.strip(), title="🚀 What's Next?", border_style="green"))


# Import domain agents
try:
    from .forge_agents import DOMAIN_AGENTS, get_agent, list_agents

    DOMAIN_AGENTS_AVAILABLE = True
except ImportError:
    DOMAIN_AGENTS = {}
    DOMAIN_AGENTS_AVAILABLE = False

# Registry of available AI agents
AI_AGENTS = {
    "copilot": CopilotAgent,
}

# Merge domain agents if available
if DOMAIN_AGENTS_AVAILABLE:
    AI_AGENTS.update(DOMAIN_AGENTS)


def register(subparsers: argparse._SubParsersAction):
    """Register the enhanced forge command with AI agent support"""
    p = subparsers.add_parser(
        COMMAND,
        help="🔨 The one command you need to know - Create FLUID data products with AI assistance",
        add_help=False,  # We'll handle help ourselves with Rich
    )

    # Custom help handler
    p.add_argument("--help", "-h", action="store_true", help="Show this help message")

    # Creation mode selection
    p.add_argument(
        "--mode",
        "-m",
        choices=[mode.value for mode in ForgeMode],
        default="copilot",
        help="Creation mode: template (traditional), copilot (AI assistant), agent (domain expert), blueprint (enterprise)",
    )

    # AI Agent selection (for agent mode)
    p.add_argument(
        "--agent",
        "-a",
        choices=list(AI_AGENTS.keys()),
        help="Specific AI agent to use (for --mode agent)",
    )

    # Traditional arguments (maintained for compatibility)
    p.add_argument("--target-dir", "-d", help="Target directory for project creation")
    p.add_argument("--template", "-t", help="Project template to use (for template mode)")
    p.add_argument("--provider", "-p", help="Infrastructure provider to use")
    p.add_argument("--blueprint", "-b", help="Blueprint to use (for blueprint mode)")

    # Workflow options
    p.add_argument(
        "--quickstart",
        "-q",
        action="store_true",
        help="Skip confirmations and use recommended defaults",
    )
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Force interactive mode even with --quickstart",
    )
    p.add_argument("--non-interactive", action="store_true", help="Use defaults without prompting")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be created without generating files",
    )

    # AI-specific options
    p.add_argument("--context", help="Additional context for AI agents (JSON string or file path)")
    p.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "claude", "gemini", "ollama"],
        help="LLM provider for copilot mode",
    )
    p.add_argument("--llm-model", help="Model identifier for copilot mode")
    p.add_argument(
        "--llm-endpoint",
        help="Exact HTTP endpoint override for the selected LLM adapter",
    )
    p.add_argument(
        "--discover",
        dest="discover",
        action="store_true",
        default=True,
        help="Inspect local files and manifests before generation",
    )
    p.add_argument(
        "--no-discover",
        dest="discover",
        action="store_false",
        help="Skip local discovery and rely only on explicit context",
    )
    p.add_argument(
        "--discovery-path",
        help="Additional local file or directory path to scan for metadata-only discovery",
    )
    p.add_argument(
        "--memory",
        dest="memory",
        action="store_true",
        default=True,
        help="Load project-scoped copilot memory when runtime/.state/copilot-memory.json exists",
    )
    p.add_argument(
        "--no-memory",
        dest="memory",
        action="store_false",
        help="Do not load project-scoped copilot memory for this run",
    )
    p.add_argument(
        "--save-memory",
        action="store_true",
        help="Persist project-scoped copilot memory after a successful non-interactive copilot run",
    )
    p.add_argument(
        "--show-memory",
        action="store_true",
        help="Show the current project-scoped copilot memory summary and exit",
    )
    p.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete the current project-scoped copilot memory file and exit",
    )
    p.add_argument(
        "--domain",
        help="Specific domain for specialized agents (e.g., finance, healthcare, retail)",
    )

    p.set_defaults(func=run)


def get_target_directory(args, default_name: str = "my-fluid-project") -> Path:
    """
    Determine target directory for project creation.
    If no target specified and we're inside the package, create outside in parent or home.
    """
    if args.target_dir:
        return Path(args.target_dir)

    # Detect if we're inside the package directory
    cwd = Path.cwd()
    try:
        package_root = Path(
            __file__
        ).parent.parent.parent  # fluid_build/cli/forge.py -> my-first-cli

        # If current directory is inside package, create in parent or home
        if cwd.is_relative_to(package_root):
            # We're inside the package, go to parent
            suggested_parent = package_root.parent
            if suggested_parent.exists() and suggested_parent.is_dir():
                return suggested_parent / default_name
            else:
                # Fallback to home directory
                return Path.home() / "fluid-projects" / default_name
    except (ValueError, Exception):
        # If path detection fails, just use current directory
        pass

    # We're outside package or detection failed, use current directory
    return cwd / default_name


def get_cli_arg(args: Any, name: str, default: Any = None) -> Any:
    """Read argparse-style attributes without letting MagicMock invent values."""
    if hasattr(args, "__dict__") and name in vars(args):
        return vars(args)[name]
    return default


def resolve_memory_store(args, logger: logging.Logger) -> CopilotMemoryStore:
    """Resolve the project-scoped memory store for management actions."""
    target_dir_value = get_cli_arg(args, "target_dir")
    target_dir = Path(target_dir_value).expanduser() if target_dir_value else None
    project_root = resolve_copilot_memory_root(Path.cwd(), target_dir=target_dir)
    return CopilotMemoryStore(project_root, logger=logger)


def handle_memory_management(args, logger: logging.Logger) -> int:
    """Show or reset project-scoped copilot memory and exit."""
    console = Console() if RICH_AVAILABLE else None
    store = resolve_memory_store(args, logger)

    if get_cli_arg(args, "reset_memory", False):
        deleted = store.delete()
        if console:
            if deleted:
                console.print(
                    f"[green]✓[/green] Deleted project-scoped copilot memory at [cyan]{store.path}[/cyan]"
                )
            else:
                console.print(
                    f"[yellow]⚠[/yellow] No project-scoped copilot memory found at [cyan]{store.path}[/cyan]"
                )
        else:
            if deleted:
                success(f"Deleted project-scoped copilot memory at {store.path}")
            else:
                warning(f"No project-scoped copilot memory found at {store.path}")
        if get_cli_arg(args, "show_memory", False):
            return handle_memory_management(
                SimpleNamespace(**{**vars(args), "reset_memory": False}),
                logger,
            )
        return 0

    memory = store.load()
    if console:
        if not memory:
            console.print(
                f"[yellow]⚠[/yellow] No project-scoped copilot memory found at [cyan]{store.path}[/cyan]"
            )
            return 0
        summary = summarize_copilot_memory(memory)
        details = [
            f"Path: `{store.path}`",
            f"Saved at: {summary.get('saved_at') or 'unknown'}",
            f"Preferred template: {summary.get('preferred_template') or 'unknown'}",
            f"Preferred provider: {summary.get('preferred_provider') or 'unknown'}",
            f"Preferred domain: {summary.get('preferred_domain') or 'unknown'}",
            f"Preferred owner: {summary.get('preferred_owner') or 'unknown'}",
            "Build engines: "
            + (", ".join(summary.get("build_engines") or []) or "none"),
            "Binding formats: "
            + (", ".join(summary.get("binding_formats") or []) or "none"),
            "Provider hints: "
            + (", ".join(summary.get("provider_hints") or []) or "none"),
            f"Schema summaries: {summary.get('schema_summary_count', 0)}",
            f"Recent successful outcomes: {summary.get('recent_outcome_count', 0)}",
        ]
        if summary.get("source_formats"):
            details.append(
                "Source formats: "
                + ", ".join(
                    f"{key}={value}" for key, value in sorted(summary["source_formats"].items())
                )
            )
        console.print(
            Panel("\n".join(details), title="🧠 Project Memory", border_style="cyan")
        )
        return 0

    if not memory:
        warning(f"No project-scoped copilot memory found at {store.path}")
        return 0
    summary = summarize_copilot_memory(memory)
    cprint(f"Project memory: {store.path}")
    cprint(f"  template={summary.get('preferred_template') or 'unknown'}")
    cprint(f"  provider={summary.get('preferred_provider') or 'unknown'}")
    cprint(f"  domain={summary.get('preferred_domain') or 'unknown'}")
    cprint(f"  owner={summary.get('preferred_owner') or 'unknown'}")
    cprint(
        "  build_engines="
        + (", ".join(summary.get("build_engines") or []) or "none")
    )
    cprint(f"  schema_summaries={summary.get('schema_summary_count', 0)}")
    cprint(f"  recent_outcomes={summary.get('recent_outcome_count', 0)}")
    return 0


def run(args, logger: logging.Logger) -> int:
    """Enhanced main entry point for forge command with AI agent support"""
    try:
        console = Console() if RICH_AVAILABLE else None

        # Handle custom help
        if hasattr(args, "help") and args.help:
            if console:
                from .help_formatter import print_forge_help

                print_forge_help()
                return 0
            else:
                # Fallback to standard help
                cprint("Run 'fluid forge' to start the interactive wizard")
                return 0

        if get_cli_arg(args, "show_memory", False) or get_cli_arg(args, "reset_memory", False):
            return handle_memory_management(args, logger)

        # Show welcome message
        if console and not args.non_interactive:
            welcome_text = """
🔨 **FLUID Forge** - The one command you need to know

Choose your creation mode:
• **copilot** - AI-powered intelligent project creation (recommended)
• **agent** - Specialized domain experts for specific industries  
• **template** - Traditional template-based creation
• **blueprint** - Complete enterprise data product templates
            """
            console.print(
                Panel(welcome_text.strip(), title="Welcome to FLUID Forge", border_style="blue")
            )

        # Determine creation mode
        mode = ForgeMode(args.mode)

        if mode == ForgeMode.AI_COPILOT:
            return run_ai_copilot_mode(args, logger)
        elif mode == ForgeMode.DOMAIN_AGENT:
            return run_domain_agent_mode(args, logger)
        elif mode == ForgeMode.BLUEPRINT:
            return run_blueprint_mode(args, logger)
        elif mode == ForgeMode.TEMPLATE:
            return run_template_mode(args, logger)
        else:
            # Default to AI Copilot for the best experience
            args.mode = "copilot"
            return run_ai_copilot_mode(args, logger)

    except Exception as e:
        logger.exception("Forge command failed")
        if console:
            console.print(f"[red]❌ Forge failed: {e}[/red]")
        else:
            console_error(f"Forge failed: {e}")
        return 1


def run_ai_copilot_mode(args, logger: logging.Logger) -> int:
    """Run forge with AI Copilot assistance"""
    console = Console() if RICH_AVAILABLE else None

    try:
        # Initialize AI Copilot
        copilot = CopilotAgent()

        if console and not args.non_interactive:
            console.print("\n[bold blue]🤖 Starting AI Copilot Assistant[/bold blue]")
            console.print(
                "[dim]I'll help you create the perfect data product by understanding your needs...[/dim]\n"
            )

        # Gather context
        context = {}

        # Load additional context if provided
        if args.context:
            try:
                loaded_context = load_context(args.context, console)
                context.update(loaded_context)
                if console:
                    console.print("[green]✓[/green] Loaded context successfully\n")
            except ContextValidationError as e:
                if console:
                    console.print(f"[red]✗[/red] Context validation failed: {e}\n")
                    console.print("[dim]Continuing without context file...[/dim]\n")
                else:
                    logger.warning(f"Context validation failed: {e}")

        # Interactive questioning if not in non-interactive mode
        if not args.non_interactive:
            context.update(gather_copilot_context(copilot, console))
        else:
            # Use defaults for non-interactive mode
            context.update(
                {
                "project_goal": "Data Analytics Platform",
                "data_sources": "Database tables",
                "use_case": "analytics",
                "complexity": "intermediate",
                }
            )

        if args.provider:
            context["provider"] = args.provider
        if args.template:
            context["template"] = args.template
        if args.domain and "domain" not in context:
            context["domain"] = args.domain

        # Determine target directory
        project_name = context.get("project_goal", "my-data-product").lower().replace(" ", "-")
        target_dir = get_target_directory(args, project_name)

        # Create project with AI assistance
        success = copilot.create_project(
            target_dir,
            context,
            {
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model,
                "llm_endpoint": args.llm_endpoint,
                "discover": args.discover,
                "discovery_path": args.discovery_path,
                "memory": args.memory,
                "save_memory": args.save_memory,
                "non_interactive": args.non_interactive,
                "target_dir": str(target_dir),
            },
            dry_run=bool(getattr(args, "dry_run", False)),
        )

        return 0 if success else 1

    except Exception as e:
        logger.exception("AI Copilot mode failed")
        if console:
            console.print(f"[red]❌ AI Copilot failed: {e}[/red]")
        return 1


def run_domain_agent_mode(args, logger: logging.Logger) -> int:
    """Run forge with specialized domain agent"""
    console = Console() if RICH_AVAILABLE else None

    try:
        # Determine which agent to use
        agent_name = args.agent

        if not agent_name:
            if console and not args.non_interactive:
                # Show available agents
                console.print("\n[bold blue]🎯 Available Domain Agents[/bold blue]")
                table = Table()
                table.add_column("Agent", style="cyan")
                table.add_column("Domain", style="green")
                table.add_column("Description", style="white")

                for name, agent_class in AI_AGENTS.items():
                    agent_instance = agent_class()
                    table.add_row(name, agent_instance.domain, agent_instance.description)

                console.print(table)

                from rich.prompt import Prompt

                agent_name = Prompt.ask(
                    "\nSelect an agent", choices=list(AI_AGENTS.keys()), default="copilot"
                )
            else:
                agent_name = "copilot"  # Default fallback

        # Initialize the selected agent
        if agent_name not in AI_AGENTS:
            if console:
                console.print(f"[red]❌ Unknown agent: {agent_name}[/red]")
                console.print(f"[dim]Available agents: {', '.join(AI_AGENTS.keys())}[/dim]")
            return 1

        agent_class = AI_AGENTS[agent_name]
        agent = agent_class()

        if console and not args.non_interactive:
            console.print(f"\n[bold blue]🎯 Starting {agent.name.title()} Domain Agent[/bold blue]")
            console.print(f"[dim]{agent.description}[/dim]\n")

        # Gather context with domain-specific questions
        context = {}

        # Load additional context if provided
        if args.context:
            try:
                from .forge_validation import validate_context_dict

                loaded_context = load_context(args.context, console)
                is_valid, error = validate_context_dict(loaded_context)
                if is_valid:
                    context.update(loaded_context)
                    if console:
                        console.print("[green]✓[/green] Loaded context successfully\n")
                else:
                    if console:
                        console.print(f"[yellow]⚠[/yellow] Context validation warning: {error}\n")
            except ContextValidationError as e:
                if console:
                    console.print(f"[red]✗[/red] Context validation failed: {e}\n")

        # Interactive questioning if not in non-interactive mode
        if not args.non_interactive:
            context.update(gather_copilot_context(agent, console))
        else:
            # Use defaults for non-interactive mode
            context = {
                "project_goal": f"{agent.domain.title()} Data Product",
                "data_sources": "Various sources",
                "use_case": "analytics",
                "complexity": "intermediate",
            }

        # Agent analyzes requirements with domain expertise
        suggestions = agent.analyze_requirements(context)

        # Show agent analysis
        if console and not args.non_interactive:
            console.print("\n[bold green]🤖 Agent Analysis Complete[/bold green]\n")
            console.print(
                f"[cyan]Recommended Template:[/cyan] {suggestions.get('recommended_template')}"
            )
            console.print(
                f"[cyan]Recommended Provider:[/cyan] {suggestions.get('recommended_provider')}"
            )
            if suggestions.get("security_requirements"):
                console.print("\n[yellow]🔒 Security Requirements:[/yellow]")
                for req in suggestions["security_requirements"][:3]:
                    console.print(f"  • {req}")
            console.print()

        # Determine target directory
        project_name = (
            context.get("project_goal", f"{agent.domain}-data-product").lower().replace(" ", "-")
        )
        from .forge_validation import sanitize_project_name

        project_name = sanitize_project_name(project_name)
        target_dir = get_target_directory(args, project_name)

        # Create project with domain agent assistance
        success = agent.create_project(target_dir, context)

        return 0 if success else 1

    except Exception as e:
        logger.exception("Domain agent mode failed")
        if console:
            console.print(f"[red]❌ Domain agent failed: {e}[/red]")
        return 1


def run_template_mode(args, logger: logging.Logger) -> int:
    """Run forge with traditional template mode"""
    console = Console() if RICH_AVAILABLE else None

    try:
        if console and not args.non_interactive:
            console.print("\n[bold blue]📋 Template Mode[/bold blue]")
            console.print("[dim]Creating project from template...[/dim]\n")

        # Use the actual ForgeEngine with proper template system
        from datetime import datetime

        from ..forge.core.engine import ForgeEngine, GenerationContext
        from ..forge.core.registry import template_registry

        template_name = args.template or "starter"
        target_dir = get_target_directory(args, f"{template_name}-project")
        provider = args.provider or "local"

        # Get template from registry
        template = template_registry.get(template_name)
        if not template:
            available = template_registry.list_available()
            logger.error(
                f"Template '{template_name}' not found. Available templates: {', '.join(available)}"
            )
            return 1

        # Create generation context
        project_config = {
            "name": target_dir.name,
            "description": f"A {template_name} data product",
            "domain": "analytics",
            "owner": "data-team",
            "provider": provider,
        }

        metadata = template.get_metadata()

        context = GenerationContext(
            project_config=project_config,
            target_dir=target_dir,
            template_metadata=metadata,
            provider_config={"provider": provider},
            user_selections={},
            forge_version="2.0.0",
            creation_time=datetime.now().isoformat(),
        )

        # Initialize ForgeEngine
        ForgeEngine()

        # Generate project
        logger.info(f"📝 Generating {template_name} project...")

        if args.dry_run if hasattr(args, "dry_run") else False:
            logger.info(f"DRY RUN: Would create project in {target_dir}")
            metadata = template.get_metadata()
            logger.info(f"Template: {metadata.name}")
            logger.info(f"Description: {metadata.description}")
            return 0

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Generate contract
        contract = template.generate_contract(context)

        # Write contract to file
        import yaml

        contract_file = target_dir / "contract.fluid.yaml"
        with open(contract_file, "w") as f:
            yaml.dump(contract, f, default_flow_style=False, sort_keys=False)

        # Generate folder structure
        structure = template.generate_structure(context)
        for path_str, content in structure.items():
            if path_str.endswith("/"):
                # Directory
                (target_dir / path_str.rstrip("/")).mkdir(parents=True, exist_ok=True)

        # Generate additional files (README, etc.)
        try:
            # Some templates have _create_readme that needs project_dir and context
            template._create_readme(target_dir, context)
        except (AttributeError, TypeError):
            # Template doesn't have README generation or has different signature
            pass

        if console:
            console.print(f"[green]✅ Template project created at {target_dir}[/green]")
        else:
            success(f"Template project created at {target_dir}")

        logger.info("\n📖 Next Steps:")
        logger.info(f"1. cd {target_dir}")
        logger.info("2. Review contract.fluid.yaml")
        logger.info("3. fluid validate contract.fluid.yaml")

        return 0

    except Exception as e:
        logger.exception("Template mode failed")
        if console:
            console.print(f"[red]❌ Template mode failed: {e}[/red]")
        else:
            console_error(f"Template mode failed: {e}")
        return 1


def gather_copilot_context(copilot: CopilotAgent, console) -> Dict[str, Any]:
    """Gather context through interactive questioning"""
    context = {}

    if not console or not RICH_AVAILABLE:
        return context

    try:
        from rich.prompt import Prompt

        questions = copilot.get_questions()

        for question_def in questions:
            key = question_def["key"]
            question = question_def["question"]
            q_type = question_def.get("type", "text")
            required = question_def.get("required", False)
            default = question_def.get("default")
            choices = question_def.get("choices")

            if q_type == "choice" and choices:
                answer = Prompt.ask(question, choices=choices, default=default)
            else:
                answer = Prompt.ask(question, default=default)

            if answer or not required:
                context[key] = answer

    except Exception:
        # Fallback to basic context if rich prompts fail
        context = {
            "project_goal": "Data Product",
            "data_sources": "Various sources",
            "use_case": "analytics",
            "complexity": "intermediate",
        }

    return context


def run_blueprint_mode(args, logger: logging.Logger) -> int:
    """Run forge with enterprise blueprint mode"""
    console = Console() if RICH_AVAILABLE else None

    try:
        if console and not args.non_interactive:
            console.print("\n[bold blue]🏗️  Blueprint Mode[/bold blue]")
            console.print("[dim]Creating enterprise data product from blueprint...[/dim]\n")

        # Get blueprint name
        blueprint_name = args.blueprint or "customer-360-gcp"

        # Get blueprint from registry
        blueprint = blueprint_registry.get_blueprint(blueprint_name)

        if not blueprint:
            available = blueprint_registry.list_blueprints()
            if console:
                console.print(f"[red]❌ Blueprint '{blueprint_name}' not found[/red]")
                console.print("\n[bold]Available blueprints:[/bold]")
                for bp in available:
                    console.print(f"  • {bp.metadata.name} - {bp.metadata.title}")
            else:
                console_error(f"Blueprint '{blueprint_name}' not found")
                cprint("\nAvailable blueprints:")
                for bp in available:
                    cprint(f"  • {bp.metadata.name} - {bp.metadata.title}")
            return 1

        # Determine target directory - create outside package
        target_dir = get_target_directory(args, blueprint_name)

        # Check if target directory exists
        if target_dir.exists() and any(target_dir.iterdir()):
            if console:
                console.print(
                    f"[yellow]⚠️  Directory {target_dir} already exists and is not empty[/yellow]"
                )
            else:
                warning(f"Directory {target_dir} already exists and is not empty")

            if not args.non_interactive:
                if console:
                    from rich.prompt import Confirm

                    if not Confirm.ask("Continue and overwrite?"):
                        return 1
                else:
                    response = input("Continue and overwrite? [y/N]: ")
                    if response.lower() != "y":
                        return 1
            else:
                return 1

        # Generate project from blueprint
        blueprint.generate_project(target_dir)

        if console:
            console.print(f"[green]✅ Blueprint project created at {target_dir}[/green]")
            console.print(
                f"[dim]{blueprint.metadata.title} - {blueprint.metadata.description}[/dim]\n"
            )

            # Show next steps
            console.print("[bold]Next steps:[/bold]")
            console.print(f"  1. cd {target_dir}")
            console.print("  2. python -m fluid_build validate contract.fluid.yaml")
            console.print(
                "  3. python -m fluid_build --provider gcp --project YOUR_PROJECT plan contract.fluid.yaml"
            )
            console.print(
                "  4. python -m fluid_build --provider gcp --project YOUR_PROJECT apply runtime/plan.json\n"
            )
        else:
            success(f"Blueprint project created at {target_dir}")
            cprint(f"{blueprint.metadata.title} - {blueprint.metadata.description}\n")
            cprint("Next steps:")
            cprint(f"  1. cd {target_dir}")
            cprint("  2. python -m fluid_build validate contract.fluid.yaml")
            cprint(
                "  3. python -m fluid_build --provider gcp --project YOUR_PROJECT plan contract.fluid.yaml"
            )
            cprint(
                "  4. python -m fluid_build --provider gcp --project YOUR_PROJECT apply runtime/plan.json\n"
            )

        return 0

    except Exception as e:
        logger.exception("Blueprint mode failed")
        if console:
            console.print(f"[red]❌ Blueprint mode failed: {e}[/red]")
        else:
            console_error(f"Blueprint mode failed: {e}")
        return 1


def load_context(context_input: str, console: Optional[Console] = None) -> Dict[str, Any]:
    """Load and validate additional context from JSON/YAML string or file

    Args:
        context_input: JSON string, file path, or YAML file path
        console: Rich console for output (optional)

    Returns:
        Validated context dictionary

    Raises:
        ContextValidationError: If context is invalid
    """
    import yaml

    try:
        # Try to parse as JSON string first
        if context_input.strip().startswith("{"):
            try:
                context = json.loads(context_input)
                if not isinstance(context, dict):
                    raise ContextValidationError("Context must be a JSON object")
                return context
            except json.JSONDecodeError as e:
                raise ContextValidationError(f"Invalid JSON: {e}")

        # Try to load as file
        context_path = Path(context_input)

        if not context_path.exists():
            raise ContextValidationError(f"Context file not found: {context_path}")

        if not context_path.is_file():
            raise ContextValidationError(f"Context path is not a file: {context_path}")

        # Check file size (max 1MB)
        if context_path.stat().st_size > 1024 * 1024:
            raise ContextValidationError("Context file too large (max 1MB)")

        # Load based on extension
        with open(context_path, encoding="utf-8") as f:
            if context_path.suffix in [".yaml", ".yml"]:
                context = yaml.safe_load(f)
            elif context_path.suffix == ".json":
                context = json.load(f)
            else:
                # Try JSON first, then YAML
                content = f.read()
                try:
                    context = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        context = yaml.safe_load(content)
                    except yaml.YAMLError as e:
                        raise ContextValidationError(f"Could not parse as JSON or YAML: {e}")

        if not isinstance(context, dict):
            raise ContextValidationError("Context must be a dictionary/object")

        # Validate context keys
        valid_keys = {
            "project_goal",
            "data_sources",
            "use_case",
            "complexity",
            "team_size",
            "domain",
            "provider",
            "owner",
            "description",
            "technologies",
        }
        invalid_keys = set(context.keys()) - valid_keys
        if invalid_keys:
            if console:
                console.print(
                    f"[yellow]Warning:[/yellow] Unknown context keys: {', '.join(invalid_keys)}"
                )

        return context

    except ContextValidationError:
        raise
    except Exception as e:
        raise ContextValidationError(f"Failed to load context: {e}")


def _run_forge_blueprint(args, blueprint_registry):
    """Run forge in blueprint mode"""
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Get the blueprint
        blueprint = blueprint_registry.get_blueprint(args.blueprint)
        if not blueprint:
            logger.error(f"Blueprint '{args.blueprint}' not found")
            logger.info("Available blueprints:")
            for bp in blueprint_registry.list_blueprints():
                logger.info(f"  - {bp.metadata.name}: {bp.metadata.title}")
            return 1

        # Determine target directory - create outside package
        target_dir = get_target_directory(args, args.blueprint)

        if target_dir.exists() and any(target_dir.iterdir()):
            if not args.non_interactive:
                response = input(
                    f"Directory {target_dir} exists and is not empty. Continue? (y/N): "
                )
                if response.lower() != "y":
                    logger.info("Operation cancelled")
                    return 1
            else:
                logger.error(f"Target directory {target_dir} exists and is not empty")
                return 1

        # Validate blueprint
        errors = blueprint.validate()
        if errors:
            logger.error("Blueprint validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1

        # Display blueprint info
        if not args.non_interactive:
            logger.info(f"📋 Blueprint: {blueprint.metadata.title}")
            logger.info(f"   Description: {blueprint.metadata.description}")
            logger.info(f"   Complexity: {blueprint.metadata.complexity.value}")
            logger.info(f"   Setup Time: {blueprint.metadata.setup_time}")
            logger.info(f"   Providers: {', '.join(blueprint.metadata.providers)}")

            if not args.quickstart:
                response = input("\nContinue with blueprint deployment? (Y/n): ")
                if response.lower() == "n":
                    logger.info("Operation cancelled")
                    return 1

        # Generate project from blueprint
        logger.info(f"🚀 Generating project from blueprint '{blueprint.metadata.name}'...")

        if args.dry_run:
            logger.info(f"DRY RUN: Would create project in {target_dir}")
            logger.info("Files that would be created:")
            for file_path in blueprint.path.rglob("*"):
                if file_path.is_file() and file_path.name != "blueprint.yaml":
                    rel_path = file_path.relative_to(blueprint.path)
                    logger.info(f"  - {rel_path}")
            return 0

        # Actually generate the project
        blueprint.generate_project(target_dir)

        logger.info(f"✅ Blueprint '{blueprint.metadata.name}' deployed successfully!")
        logger.info(f"📁 Project created in: {target_dir}")

        # Show next steps
        logger.info("\n📖 Next Steps:")
        logger.info(f"1. cd {target_dir}")
        logger.info("2. Review the generated files and documentation")
        logger.info("3. Configure your data sources")
        logger.info("4. Run: fluid validate")
        logger.info("5. Run: dbt run (if using dbt)")

        return 0

    except Exception as e:
        logger.error(f"Blueprint deployment failed: {e}", exc_info=True)
        return 1


# Backward compatibility functions for migration period
def create_legacy_bootstrapper(target_dir: Optional[str] = None, **kwargs):
    """
    Create a legacy bootstrapper for backward compatibility

    This function allows existing code that depends on ForgeBootstrapper
    to continue working during the migration period.
    """
    from .forge_legacy import ForgeBootstrapper

    return ForgeBootstrapper(target_dir, **kwargs)


def get_enhanced_templates():
    """
    Get enhanced templates for backward compatibility

    This function maps the old template format to the new registry system.
    """
    from ..forge.core.registry import template_registry

    # Convert new template system to legacy format
    legacy_templates = {}

    for template_name in template_registry.list_available():
        template = template_registry.get(template_name)
        if template:
            metadata = template.get_metadata()

            # Create legacy-compatible template structure
            legacy_templates[template_name] = {
                "name": metadata.name,
                "description": metadata.description,
                "complexity": metadata.complexity.value,
                "provider_support": metadata.provider_support,
                "use_cases": metadata.use_cases,
                "technologies": metadata.technologies,
                "estimated_time": metadata.estimated_time,
                "tags": metadata.tags,
            }

    return legacy_templates


# Export compatibility layer
__all__ = ["register", "run", "create_legacy_bootstrapper", "get_enhanced_templates"]
