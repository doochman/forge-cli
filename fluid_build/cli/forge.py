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

"""Public Forge CLI entrypoint and compatibility surface."""

from __future__ import annotations

import argparse
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fluid_build.cli.console import cprint
from fluid_build.cli.console import error as console_error
from fluid_build.cli.forge_agents import DOMAIN_AGENTS
from fluid_build.cli.forge_context import (
    gather_copilot_context as _gather_context,
)
from fluid_build.cli.forge_context import (
    get_cli_arg as _get_cli_arg,
)
from fluid_build.cli.forge_context import (
    get_target_directory as _get_target_dir,
)
from fluid_build.cli.forge_context import (
    handle_memory_management as _handle_memory,
)
from fluid_build.cli.forge_context import (
    load_context as _load_ctx,
)
from fluid_build.cli.forge_context import (
    resolve_memory_store as _resolve_store,
)
from fluid_build.cli.forge_copilot_agent import (
    AIAgent,
    CopilotAgentBase,
    recommend_template_for_use_case,
)
from fluid_build.cli.forge_copilot_interview import build_interview_summary_from_context
from fluid_build.cli.forge_copilot_memory import (
    CopilotMemoryStore,
    resolve_copilot_memory_root,
    summarize_copilot_memory,
)
from fluid_build.cli.forge_copilot_runtime import (
    CopilotGenerationError,
    CopilotGenerationResult,
    build_capability_matrix,
    discover_local_context,
    generate_copilot_artifacts,
    normalize_provider_name,
    normalize_template_name,
    resolve_llm_config,
)
from fluid_build.cli.forge_copilot_taxonomy import normalize_copilot_context
from fluid_build.cli.forge_dialogs import ask_confirmation
from fluid_build.cli.forge_modes import (
    run_ai_copilot_mode as _run_copilot,
)
from fluid_build.cli.forge_modes import (
    run_blueprint_mode as _run_blueprint,
)
from fluid_build.cli.forge_modes import (
    run_domain_agent_mode as _run_agent,
)
from fluid_build.cli.forge_modes import (
    run_forge_blueprint_impl as _run_blueprint_legacy,
)
from fluid_build.cli.forge_modes import (
    run_template_mode as _run_template,
)
from fluid_build.cli.forge_ui import print_welcome_panel

try:
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised through non-Rich fallbacks
    Console = None  # type: ignore[assignment]
    RICH_AVAILABLE = False

from ..blueprints import registry as blueprint_registry
from ._common import CLIError

COMMAND = "forge"
LOG = logging.getLogger("fluid.cli.forge")


class ForgeError(CLIError):
    """Base exception for Forge command errors."""


class TemplateNotFoundError(ForgeError):
    """Template not found in registry."""

    def __init__(self, template_name: str, available: List[str]):
        self.template_name = template_name
        self.available = available
        super().__init__(f"Template '{template_name}' not found. Available: {', '.join(available)}")


class BlueprintNotFoundError(ForgeError):
    """Blueprint not found in registry."""

    def __init__(self, blueprint_name: str, available: List[str]):
        self.blueprint_name = blueprint_name
        self.available = available
        super().__init__(
            f"Blueprint '{blueprint_name}' not found. Available: {', '.join(available)}"
        )


class InvalidProjectNameError(ForgeError):
    """Invalid project name format."""

    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason
        super().__init__(f"Invalid project name '{name}': {reason}")


class ProjectGenerationError(ForgeError):
    """Project generation failed."""


class ContextValidationError(ForgeError):
    """Context file validation failed."""


class ForgeMode(Enum):
    """Forge creation modes."""

    TEMPLATE = "template"
    AI_COPILOT = "copilot"
    DOMAIN_AGENT = "agent"
    BLUEPRINT = "blueprint"


class CopilotAgent(CopilotAgentBase):
    """Public copilot agent wired to the compatibility aliases in this module."""

    def _resolve_llm_config_dependency(self, options):
        return resolve_llm_config(options)

    def _discover_local_context_dependency(self, options):
        return discover_local_context(
            getattr(options, "discovery_path", None),
            discover=getattr(options, "discover", True),
            workspace_root=Path.cwd(),
            logger=LOG,
        )

    def _build_capability_matrix_dependency(self):
        return build_capability_matrix()

    def _generate_copilot_artifacts_dependency(
        self,
        context: Dict[str, Any],
        *,
        llm_config: Any,
        discovery_report: Any,
        project_memory: Any,
        capability_matrix: Any,
    ) -> CopilotGenerationResult:
        return generate_copilot_artifacts(
            context,
            llm_config=llm_config,
            discovery_report=discovery_report,
            project_memory=project_memory,
            capability_matrix=capability_matrix,
            logger=LOG,
        )

    def _make_memory_store_dependency(self, project_root: Path) -> CopilotMemoryStore:
        return CopilotMemoryStore(project_root, logger=LOG)

    def _ask_confirmation_dependency(self, prompt: str, preview: str) -> bool:
        if self.console and RICH_AVAILABLE:
            return ask_confirmation(
                self.console,
                prompt,
                default=False,
                title="🧠 Save Project Memory?",
                preview=preview,
                border_style="cyan",
            )
        return super()._ask_confirmation_dependency(prompt, preview)


DOMAIN_AGENTS_AVAILABLE = bool(DOMAIN_AGENTS)
AI_AGENTS = {"copilot": CopilotAgent}
if DOMAIN_AGENTS_AVAILABLE:
    AI_AGENTS.update(DOMAIN_AGENTS)


def register(subparsers: argparse._SubParsersAction):
    """Register the Forge command with AI agent support."""
    parser = subparsers.add_parser(
        COMMAND,
        help="🔨 The one command you need to know - Create FLUID data products with AI assistance",
        add_help=False,
    )
    parser.add_argument("--help", "-h", action="store_true", help="Show this help message")
    parser.add_argument(
        "--mode",
        "-m",
        choices=[mode.value for mode in ForgeMode],
        default="copilot",
        help="Creation mode: template (traditional), copilot (AI assistant), agent (domain expert), blueprint (enterprise)",
    )
    parser.add_argument(
        "--agent",
        "-a",
        choices=list(AI_AGENTS.keys()),
        help="Specific AI agent to use (for --mode agent)",
    )
    parser.add_argument("--target-dir", "-d", help="Target directory for project creation")
    parser.add_argument("--template", "-t", help="Project template to use (for template mode)")
    parser.add_argument("--provider", "-p", help="Infrastructure provider to use")
    parser.add_argument("--blueprint", "-b", help="Blueprint to use (for blueprint mode)")
    parser.add_argument(
        "--quickstart",
        "-q",
        action="store_true",
        help="Skip confirmations and use recommended defaults",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Force interactive mode even with --quickstart",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use defaults without prompting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be created without generating files",
    )
    parser.add_argument(
        "--context", help="Additional context for AI agents (JSON string or file path)"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "claude", "gemini", "ollama"],
        help="LLM provider for copilot mode",
    )
    parser.add_argument("--llm-model", help="Model identifier for copilot mode")
    parser.add_argument(
        "--llm-endpoint",
        help="Exact HTTP endpoint override for the selected LLM adapter",
    )
    parser.add_argument(
        "--discover",
        dest="discover",
        action="store_true",
        default=True,
        help="Inspect local files and manifests before generation",
    )
    parser.add_argument(
        "--no-discover",
        dest="discover",
        action="store_false",
        help="Skip local discovery and rely only on explicit context",
    )
    parser.add_argument(
        "--discovery-path",
        help="Additional local file or directory path to scan for metadata-only discovery",
    )
    parser.add_argument(
        "--memory",
        dest="memory",
        action="store_true",
        default=True,
        help="Load project-scoped copilot memory when runtime/.state/copilot-memory.json exists",
    )
    parser.add_argument(
        "--no-memory",
        dest="memory",
        action="store_false",
        help="Do not load project-scoped copilot memory for this run",
    )
    parser.add_argument(
        "--save-memory",
        action="store_true",
        help="Persist project-scoped copilot memory after a successful non-interactive copilot run",
    )
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="Show the current project-scoped copilot memory summary and exit",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete the current project-scoped copilot memory file and exit",
    )
    parser.add_argument(
        "--domain",
        help="Specific domain for specialized agents (e.g., finance, healthcare, retail, telco)",
    )
    parser.set_defaults(func=run)


def get_target_directory(args, default_name: str = "my-fluid-project") -> Path:
    return _get_target_dir(args, default_name)


def get_cli_arg(args: Any, name: str, default: Any = None) -> Any:
    return _get_cli_arg(args, name, default)


def resolve_memory_store(args, logger: logging.Logger) -> CopilotMemoryStore:
    return _resolve_store(args, logger, memory_store_class=CopilotMemoryStore)


def handle_memory_management(args, logger: logging.Logger) -> int:
    return _handle_memory(
        args,
        logger,
        memory_store_class=CopilotMemoryStore,
        console_factory=Console if RICH_AVAILABLE else None,
    )


def run(args, logger: logging.Logger) -> int:
    """Enhanced main entry point for forge command with AI agent support."""
    try:
        console = Console() if RICH_AVAILABLE else None
        if hasattr(args, "help") and args.help:
            if console:
                from .help_formatter import print_forge_help

                print_forge_help()
                return 0
            cprint("Run 'fluid forge' to start the interactive wizard")
            return 0

        if get_cli_arg(args, "show_memory", False) or get_cli_arg(args, "reset_memory", False):
            return handle_memory_management(args, logger)

        if console and not args.non_interactive:
            print_welcome_panel(console)

        mode = ForgeMode(args.mode)
        if mode == ForgeMode.AI_COPILOT:
            return run_ai_copilot_mode(args, logger)
        if mode == ForgeMode.DOMAIN_AGENT:
            return run_domain_agent_mode(args, logger)
        if mode == ForgeMode.BLUEPRINT:
            return run_blueprint_mode(args, logger)
        if mode == ForgeMode.TEMPLATE:
            return run_template_mode(args, logger)

        args.mode = "copilot"
        return run_ai_copilot_mode(args, logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Forge command failed")
        if "console" in locals() and console:
            console.print(f"[red]❌ Forge failed: {exc}[/red]")
        else:
            console_error(f"Forge failed: {exc}")
        return 1


def run_ai_copilot_mode(args, logger: logging.Logger) -> int:
    return _run_copilot(
        args,
        logger,
        copilot_class=CopilotAgent,
        get_cli_arg_fn=get_cli_arg,
        load_context_fn=load_context,
        get_target_directory_fn=get_target_directory,
        context_error_cls=ContextValidationError,
        build_interview_summary_fn=build_interview_summary_from_context,
        console_factory=Console if RICH_AVAILABLE else None,
    )


def run_domain_agent_mode(args, logger: logging.Logger) -> int:
    return _run_agent(
        args,
        logger,
        ai_agents=AI_AGENTS,
        gather_context_fn=gather_copilot_context,
        load_context_fn=load_context,
        get_target_directory_fn=get_target_directory,
        context_error_cls=ContextValidationError,
        console_factory=Console if RICH_AVAILABLE else None,
    )


def run_template_mode(args, logger: logging.Logger) -> int:
    return _run_template(
        args,
        logger,
        get_target_directory_fn=get_target_directory,
        console_factory=Console if RICH_AVAILABLE else None,
    )


def gather_copilot_context(copilot: CopilotAgent, console) -> Dict[str, Any]:
    return _gather_context(copilot, console)


def run_blueprint_mode(args, logger: logging.Logger) -> int:
    return _run_blueprint(
        args,
        logger,
        blueprint_registry=blueprint_registry,
        get_target_directory_fn=get_target_directory,
        ask_confirmation_fn=ask_confirmation,
        console_factory=Console if RICH_AVAILABLE else None,
    )


def load_context(
    context_input: str,
    console: Optional[Any] = None,
    *,
    context_error_cls: type[Exception] = ContextValidationError,
) -> Dict[str, Any]:
    return _load_ctx(
        context_input,
        console,
        context_error_cls=context_error_cls,
    )


def _run_forge_blueprint(args, blueprint_registry):
    return _run_blueprint_legacy(
        args,
        blueprint_registry,
        get_target_directory_fn=get_target_directory,
    )


def create_legacy_bootstrapper(target_dir: Optional[str] = None, **kwargs):
    """Create a legacy bootstrapper for backward compatibility."""
    from .forge_legacy import ForgeBootstrapper

    return ForgeBootstrapper(target_dir, **kwargs)


def get_enhanced_templates():
    """Get enhanced templates for backward compatibility."""
    from ..forge.core.registry import template_registry

    legacy_templates = {}
    for template_name in template_registry.list_available():
        template = template_registry.get(template_name)
        if template:
            metadata = template.get_metadata()
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


__all__ = [
    "AIAgent",
    "AI_AGENTS",
    "BlueprintNotFoundError",
    "COMMAND",
    "ContextValidationError",
    "CopilotAgent",
    "ForgeError",
    "ForgeMode",
    "InvalidProjectNameError",
    "ProjectGenerationError",
    "TemplateNotFoundError",
    "create_legacy_bootstrapper",
    "gather_copilot_context",
    "get_cli_arg",
    "get_enhanced_templates",
    "get_target_directory",
    "handle_memory_management",
    "load_context",
    "register",
    "resolve_memory_store",
    "run",
    "run_ai_copilot_mode",
    "run_blueprint_mode",
    "run_domain_agent_mode",
    "run_template_mode",
]
