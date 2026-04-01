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

"""Mode handlers for `fluid forge`."""

from __future__ import annotations

__all__ = [
    "run_ai_copilot_mode",
    "run_blueprint_mode",
    "run_domain_agent_mode",
    "run_forge_blueprint_impl",
    "run_template_mode",
]


import logging
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from fluid_build.cli.console import cprint, success, warning
from fluid_build.cli.console import error as console_error
from fluid_build.cli.forge_copilot_interview import run_adaptive_copilot_interview
from fluid_build.cli.forge_copilot_taxonomy import normalize_copilot_context
from fluid_build.cli.forge_dialogs import ask_confirmation, print_dialog_status
from fluid_build.cli.forge_ui import print_assumptions_panel, show_blueprint_next_steps

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via non-Rich fallbacks
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    RICH_AVAILABLE = False


def run_ai_copilot_mode(
    args: Any,
    logger: logging.Logger,
    *,
    copilot_class: type,
    get_cli_arg_fn: Callable[[Any, str, Any], Any],
    load_context_fn: Callable[..., Dict[str, Any]],
    get_target_directory_fn: Callable[[Any, str], Path],
    context_error_cls: type[Exception],
    build_interview_summary_fn: Callable[[Mapping[str, Any]], Dict[str, Any]],
    console_factory: Optional[Callable[[], Any]] = Console if RICH_AVAILABLE else None,
) -> int:
    """Run Forge with AI copilot assistance."""
    console = console_factory() if console_factory else None

    try:
        copilot = copilot_class()

        if console and not args.non_interactive:
            console.print("\n[bold blue]🤖 Starting AI Copilot Assistant[/bold blue]")
            console.print(
                "[dim]I'll help you create the perfect data product by understanding your needs...[/dim]\n"
            )

        context: Dict[str, Any] = {}
        copilot_options = {
            "llm_provider": get_cli_arg_fn(args, "llm_provider"),
            "llm_model": get_cli_arg_fn(args, "llm_model"),
            "llm_endpoint": get_cli_arg_fn(args, "llm_endpoint"),
            "discover": get_cli_arg_fn(args, "discover", True),
            "discovery_path": get_cli_arg_fn(args, "discovery_path"),
            "memory": get_cli_arg_fn(args, "memory", True),
            "save_memory": get_cli_arg_fn(args, "save_memory", False),
            "non_interactive": get_cli_arg_fn(args, "non_interactive", False),
        }

        context_arg = get_cli_arg_fn(args, "context")
        if context_arg:
            try:
                loaded_context = load_context_fn(
                    context_arg,
                    console,
                    context_error_cls=context_error_cls,
                )
                context.update(loaded_context)
                if console:
                    print_dialog_status(console, status="success", message="Loaded extra context.")
            except context_error_cls as exc:
                if console:
                    print_dialog_status(
                        console,
                        status="error",
                        message=f"Couldn't use the context file: {exc}",
                        detail="Continuing without it for now.",
                    )
                else:
                    logger.warning("Context validation failed: %s", exc)

        if get_cli_arg_fn(args, "provider"):
            context["provider"] = get_cli_arg_fn(args, "provider")
        if get_cli_arg_fn(args, "template"):
            context["template"] = get_cli_arg_fn(args, "template")
        if get_cli_arg_fn(args, "domain") and "domain" not in context:
            context["domain"] = get_cli_arg_fn(args, "domain")
        explicit_target_dir = get_cli_arg_fn(args, "target_dir")
        if explicit_target_dir:
            copilot_options["target_dir"] = str(Path(explicit_target_dir).expanduser())

        if not get_cli_arg_fn(args, "non_interactive", False):
            runtime_inputs = copilot.prepare_runtime_inputs(copilot_options)
            copilot_options.update(runtime_inputs)
            capability_warnings = list(runtime_inputs.get("capability_warnings") or [])
            if console and capability_warnings:
                print_dialog_status(
                    console,
                    status="warning",
                    message="Copilot couldn't fully verify some local providers.",
                    detail=(
                        f"{capability_warnings[0]} "
                        "Continuing with best-effort defaults. You can review or override the provider later."
                    ),
                )
            interview_state = run_adaptive_copilot_interview(
                initial_context=context,
                console=console,
                llm_config=runtime_inputs["llm_config"],
                discovery_report=runtime_inputs["discovery_report"],
                capability_matrix=runtime_inputs["capability_matrix"],
                project_memory=runtime_inputs["project_memory"],
            )
            copilot_options["interview_state"] = interview_state
            context = interview_state.finalize()
            assumptions = list(context.get("assumptions_used") or [])
            if console and assumptions:
                print_assumptions_panel(console, assumptions)
        else:
            for key, value in {
                "project_goal": "Data Analytics Platform",
                "data_sources": "Database tables",
                "use_case": "analytics",
                "complexity": "intermediate",
            }.items():
                context.setdefault(key, value)
            context["interview_summary"] = build_interview_summary_fn(context)

        context = normalize_copilot_context(context)
        project_name = context.get("project_goal", "my-data-product").lower().replace(" ", "-")
        target_dir = get_target_directory_fn(args, project_name)
        copilot_options["target_dir"] = str(target_dir)

        success_result = copilot.create_project(
            target_dir,
            context,
            copilot_options,
            dry_run=bool(get_cli_arg_fn(args, "dry_run", False)),
        )
        return 0 if success_result else 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("AI Copilot mode failed")
        if console:
            console.print(f"[red]❌ AI Copilot failed: {exc}[/red]")
        return 1


def run_domain_agent_mode(
    args: Any,
    logger: logging.Logger,
    *,
    ai_agents: Mapping[str, type],
    gather_context_fn: Callable[[Any, Any], Dict[str, Any]],
    load_context_fn: Callable[..., Dict[str, Any]],
    get_target_directory_fn: Callable[[Any, str], Path],
    context_error_cls: type[Exception],
    console_factory: Optional[Callable[[], Any]] = Console if RICH_AVAILABLE else None,
) -> int:
    """Run Forge with a specialized domain agent."""
    console = console_factory() if console_factory else None

    try:
        agent_name = args.agent

        if not agent_name:
            if console and not args.non_interactive and Table is not None:
                console.print("\n[bold blue]🎯 Available Domain Agents[/bold blue]")
                table = Table()
                table.add_column("Agent", style="cyan")
                table.add_column("Domain", style="green")
                table.add_column("Description", style="white")
                for name, agent_class in ai_agents.items():
                    agent_instance = agent_class()
                    table.add_row(name, agent_instance.domain, agent_instance.description)
                console.print(table)

                from fluid_build.cli.forge_copilot_interview import InterviewQuestion
                from fluid_build.cli.forge_dialogs import ask_dialog_question

                selection = ask_dialog_question(
                    console,
                    InterviewQuestion(
                        id="agent",
                        field="agent",
                        prompt="Which agent would you like to use?",
                        type="choice",
                        choices=[{"label": name, "value": name} for name in ai_agents.keys()],
                        required=False,
                        allow_skip=True,
                        default="copilot",
                    ),
                )
                agent_name = selection.value or "copilot"
            else:
                agent_name = "copilot"

        if agent_name not in ai_agents:
            if console:
                console.print(f"[red]❌ Unknown agent: {agent_name}[/red]")
                console.print(f"[dim]Available agents: {', '.join(ai_agents.keys())}[/dim]")
            return 1

        agent = ai_agents[agent_name]()

        if console and not args.non_interactive:
            console.print(f"\n[bold blue]🎯 Starting {agent.name.title()} Domain Agent[/bold blue]")
            console.print(f"[dim]{agent.description}[/dim]\n")

        context: Dict[str, Any] = {}
        if args.context:
            try:
                from fluid_build.cli.forge_validation import validate_context_dict

                loaded_context = load_context_fn(
                    args.context,
                    console,
                    context_error_cls=context_error_cls,
                )
                is_valid, error = validate_context_dict(loaded_context)
                if is_valid:
                    context.update(loaded_context)
                    if console:
                        print_dialog_status(
                            console, status="success", message="Loaded extra context."
                        )
                elif console:
                    print_dialog_status(
                        console,
                        status="warning",
                        message=f"Context loaded with a warning: {error}",
                    )
            except context_error_cls as exc:
                if console:
                    print_dialog_status(
                        console,
                        status="error",
                        message=f"Couldn't use the context file: {exc}",
                    )

        if not args.non_interactive:
            context.update(gather_context_fn(agent, console))
        else:
            context = {
                "project_goal": f"{agent.domain.title()} Data Product",
                "data_sources": "Various sources",
                "use_case": "analytics",
                "complexity": "intermediate",
            }

        suggestions = agent.analyze_requirements(context)
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
                for requirement in suggestions["security_requirements"][:3]:
                    console.print(f"  • {requirement}")
            console.print()

        project_name = (
            context.get("project_goal", f"{agent.domain}-data-product").lower().replace(" ", "-")
        )
        from fluid_build.cli.forge_validation import sanitize_project_name

        target_dir = get_target_directory_fn(args, sanitize_project_name(project_name))
        success_result = agent.create_project(target_dir, context)
        return 0 if success_result else 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Domain agent mode failed")
        if console:
            console.print(f"[red]❌ Domain agent failed: {exc}[/red]")
        return 1


def run_template_mode(
    args: Any,
    logger: logging.Logger,
    *,
    get_target_directory_fn: Callable[[Any, str], Path],
    console_factory: Optional[Callable[[], Any]] = Console if RICH_AVAILABLE else None,
) -> int:
    """Run Forge with traditional template mode."""
    console = console_factory() if console_factory else None

    try:
        if console and not args.non_interactive:
            console.print("\n[bold blue]📋 Template Mode[/bold blue]")
            console.print("[dim]Creating project from template...[/dim]\n")

        from datetime import datetime

        from fluid_build.forge.core.engine import ForgeEngine, GenerationContext
        from fluid_build.forge.core.registry import template_registry

        template_name = args.template or "starter"
        target_dir = get_target_directory_fn(args, f"{template_name}-project")
        provider = args.provider or "local"
        template = template_registry.get(template_name)
        if not template:
            available = template_registry.list_available()
            logger.error(
                "Template '%s' not found. Available templates: %s",
                template_name,
                ", ".join(available),
            )
            return 1

        metadata = template.get_metadata()
        context = GenerationContext(
            project_config={
                "name": target_dir.name,
                "description": f"A {template_name} data product",
                "domain": "analytics",
                "owner": "data-team",
                "provider": provider,
            },
            target_dir=target_dir,
            template_metadata=metadata,
            provider_config={"provider": provider},
            user_selections={},
            forge_version="2.0.0",
            creation_time=datetime.now().isoformat(),
        )

        ForgeEngine()
        logger.info("📝 Generating %s project...", template_name)

        if args.dry_run if hasattr(args, "dry_run") else False:
            logger.info("DRY RUN: Would create project in %s", target_dir)
            logger.info("Template: %s", metadata.name)
            logger.info("Description: %s", metadata.description)
            return 0

        target_dir.mkdir(parents=True, exist_ok=True)
        contract = template.generate_contract(context)
        import yaml

        with open(target_dir / "contract.fluid.yaml", "w") as handle:
            yaml.dump(contract, handle, default_flow_style=False, sort_keys=False)

        for path_str, content in template.generate_structure(context).items():
            if path_str.endswith("/"):
                (target_dir / path_str.rstrip("/")).mkdir(parents=True, exist_ok=True)

        try:
            template._create_readme(target_dir, context)
        except (AttributeError, TypeError):
            pass

        if console:
            console.print(f"[green]✅ Template project created at {target_dir}[/green]")
        else:
            success(f"Template project created at {target_dir}")

        logger.info("\n📖 Next Steps:")
        logger.info("1. cd %s", target_dir)
        logger.info("2. Review contract.fluid.yaml")
        logger.info("3. fluid validate contract.fluid.yaml")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Template mode failed")
        if console:
            console.print(f"[red]❌ Template mode failed: {exc}[/red]")
        else:
            console_error(f"Template mode failed: {exc}")
        return 1


def run_blueprint_mode(
    args: Any,
    logger: logging.Logger,
    *,
    blueprint_registry: Any,
    get_target_directory_fn: Callable[[Any, str], Path],
    ask_confirmation_fn: Callable[..., bool] = ask_confirmation,
    console_factory: Optional[Callable[[], Any]] = Console if RICH_AVAILABLE else None,
) -> int:
    """Run Forge with enterprise blueprint mode."""
    console = console_factory() if console_factory else None

    try:
        if console and not args.non_interactive:
            console.print("\n[bold blue]🏗️  Blueprint Mode[/bold blue]")
            console.print("[dim]Creating enterprise data product from blueprint...[/dim]\n")

        blueprint_name = args.blueprint or "customer-360-gcp"
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

        target_dir = get_target_directory_fn(args, blueprint_name)
        if target_dir.exists() and any(target_dir.iterdir()):
            if console:
                console.print(
                    f"[yellow]⚠️  Directory {target_dir} already exists and is not empty[/yellow]"
                )
            else:
                warning(f"Directory {target_dir} already exists and is not empty")

            if not args.non_interactive:
                if console:
                    if not ask_confirmation_fn(
                        console,
                        "Continue and overwrite the existing directory?",
                        default=False,
                    ):
                        return 1
                else:
                    response = input("Continue and overwrite? [y/N]: ")
                    if response.lower() != "y":
                        return 1
            else:
                return 1

        blueprint.generate_project(target_dir)

        if console:
            console.print(f"[green]✅ Blueprint project created at {target_dir}[/green]")
            console.print(
                f"[dim]{blueprint.metadata.title} - {blueprint.metadata.description}[/dim]\n"
            )
            show_blueprint_next_steps(console)
        else:
            success(f"Blueprint project created at {target_dir}")
            cprint(f"{blueprint.metadata.title} - {blueprint.metadata.description}\n")
            cprint("Next steps:")
            cprint("  1. fluid validate contract.fluid.yaml")
            cprint("  2. fluid plan contract.fluid.yaml --out runtime/plan.json")
            cprint("  3. fluid apply runtime/plan.json\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Blueprint mode failed")
        if console:
            console.print(f"[red]❌ Blueprint mode failed: {exc}[/red]")
        else:
            console_error(f"Blueprint mode failed: {exc}")
        return 1


def run_forge_blueprint_impl(
    args: Any,
    blueprint_registry: Any,
    *,
    get_target_directory_fn: Callable[[Any, str], Path],
) -> int:
    """Legacy blueprint execution path retained for compatibility."""
    logger = logging.getLogger(__name__)

    try:
        blueprint = blueprint_registry.get_blueprint(args.blueprint)
        if not blueprint:
            logger.error("Blueprint '%s' not found", args.blueprint)
            logger.info("Available blueprints:")
            for bp in blueprint_registry.list_blueprints():
                logger.info("  - %s: %s", bp.metadata.name, bp.metadata.title)
            return 1

        target_dir = get_target_directory_fn(args, args.blueprint)
        if target_dir.exists() and any(target_dir.iterdir()):
            if not args.non_interactive:
                response = input(
                    f"Directory {target_dir} exists and is not empty. Continue? (y/N): "
                )
                if response.lower() != "y":
                    logger.info("Operation cancelled")
                    return 1
            else:
                logger.error("Target directory %s exists and is not empty", target_dir)
                return 1

        errors = blueprint.validate()
        if errors:
            logger.error("Blueprint validation failed:")
            for error in errors:
                logger.error("  - %s", error)
            return 1

        if not args.non_interactive:
            logger.info("📋 Blueprint: %s", blueprint.metadata.title)
            logger.info("   Description: %s", blueprint.metadata.description)
            logger.info("   Complexity: %s", blueprint.metadata.complexity.value)
            logger.info("   Setup Time: %s", blueprint.metadata.setup_time)
            logger.info("   Providers: %s", ", ".join(blueprint.metadata.providers))
            if not args.quickstart:
                response = input("\nContinue with blueprint deployment? (Y/n): ")
                if response.lower() == "n":
                    logger.info("Operation cancelled")
                    return 1

        logger.info("🚀 Generating project from blueprint '%s'...", blueprint.metadata.name)
        if args.dry_run:
            logger.info("DRY RUN: Would create project in %s", target_dir)
            logger.info("Files that would be created:")
            for file_path in blueprint.path.rglob("*"):
                if file_path.is_file() and file_path.name != "blueprint.yaml":
                    logger.info("  - %s", file_path.relative_to(blueprint.path))
            return 0

        blueprint.generate_project(target_dir)
        logger.info("✅ Blueprint '%s' deployed successfully!", blueprint.metadata.name)
        logger.info("📁 Project created in: %s", target_dir)
        logger.info("\n📖 Next Steps:")
        logger.info("1. cd %s", target_dir)
        logger.info("2. Review the generated files and documentation")
        logger.info("3. Configure your data sources")
        logger.info("4. Run: fluid validate")
        logger.info("5. Run: dbt run (if using dbt)")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Blueprint deployment failed: %s", exc, exc_info=True)
        return 1
