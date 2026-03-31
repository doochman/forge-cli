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
Custom Rich-formatted help for FLUID CLI

Provides beautiful, colorful help output using Rich library.
"""

from __future__ import annotations

import argparse

from fluid_build import __version__ as _VERSION

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def print_first_run_help(parser: argparse.ArgumentParser) -> None:
    """
    Compact first-run experience for new users.

    Shown when ~/.fluid does not exist yet.  Focused: three steps, one
    example command, and a pointer to --help for the full reference.
    """
    if not RICH_AVAILABLE:
        parser.print_help()
        return

    console = Console()
    console.print()

    # Welcome — warm but brief
    console.print(
        Panel(
            "[bold bright_cyan]🌊 Welcome to FLUID Forge![/bold bright_cyan]\n\n"
            "[bright_white]Build, deploy and govern data products with declarative contracts.[/bright_white]\n"
            "[dim]No cloud account required — the local provider runs everything on your machine.[/dim]",
            border_style="bright_cyan",
            padding=(1, 2),
            title="[bold bright_white]👋 First Time?[/bold bright_white]",
            title_align="left",
        )
    )
    console.print()

    # Three-step quick start
    console.print("[bold bright_green]Get started in 60 seconds:[/bold bright_green]\n")

    steps = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    steps.add_column(style="bold bright_yellow", width=4)
    steps.add_column(style="bright_cyan", width=46)
    steps.add_column(style="dim bright_white")

    steps.add_row(
        "1.", "fluid init my-project --quickstart", "Create a working project with sample data"
    )
    steps.add_row(
        "2.", "cd my-project && fluid validate contract.fluid.yaml", "Check the generated contract"
    )
    steps.add_row("3.", "fluid apply contract.fluid.yaml --yes", "Run the pipeline end-to-end")

    console.print(steps)
    console.print()

    # Helpful pointers
    console.print(
        Panel(
            "[bright_yellow]fluid doctor[/bright_yellow]       Check your system is ready\n"
            "[bright_yellow]fluid --help[/bright_yellow]       See all commands & options\n"
            "[bright_yellow]fluid <cmd> -h[/bright_yellow]     Help for a specific command\n\n"
            "[dim]📚 Docs:[/dim]  [bright_cyan]https://github.com/open-data-protocol/fluid[/bright_cyan]",
            title="[bold bright_white]What's next?[/bold bright_white]",
            title_align="left",
            border_style="bright_white",
            padding=(1, 2),
        )
    )
    console.print()


def print_main_help(parser: argparse.ArgumentParser) -> None:
    """
    Print polished main help message with Rich formatting.

    Shows the core workflow, generation, enterprise integration,
    quality/governance, and utilities — plus a discovery line for
    the remaining power-user commands.
    """
    if not RICH_AVAILABLE:
        parser.print_help()
        return

    console = Console()
    W = console.width or 80  # terminal width
    console.print()

    # ── Header ──────────────────────────────────────────────────────
    bar = "[dim bright_cyan]" + "━" * (W - 4) + "[/dim bright_cyan]"
    console.print(f"  {bar}")
    console.print(
        "  [bold bright_cyan]🌊  F L U I D   F O R G E[/bold bright_cyan]  "
        f"[dim]v{_VERSION}[/dim]     "
        "[italic bright_white]Declarative data products — from contract to cloud[/italic bright_white]"
    )
    console.print(f"  {bar}")
    console.print()

    # ── Usage ───────────────────────────────────────────────────────
    console.print(
        "  [bold]USAGE[/bold]   "
        "[bright_cyan]fluid[/bright_cyan] "
        "[bright_green]<command>[/bright_green] [dim][options][/dim]        "
        "[dim]Run[/dim] [bright_cyan]fluid <cmd> -h[/bright_cyan] [dim]for per-command help[/dim]"
    )
    console.print()

    # ── helper to build a group ─────────────────────────────────────
    def _section(icon: str, title: str, color: str, rows: list[tuple[str, str]]) -> None:
        tbl = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            pad_edge=False,
        )
        tbl.add_column(style=f"{color} bold", min_width=24, max_width=24)
        tbl.add_column(style="bright_white")
        for cmd, desc in rows:
            tbl.add_row(f"  {cmd}", desc)
        console.print(f"  {icon} [bold {color}]{title}[/bold {color}]")
        console.print(tbl)
        console.print()

    # ── Core Workflow ───────────────────────────────────────────────
    _section(
        "▸",
        "Core Workflow",
        "bright_blue",
        [
            ("init", "Create a new project  [dim]--quickstart · --scan · --wizard[/dim]"),
            ("validate", "Check contract syntax and provider rules"),
            ("plan", "Preview what will change  [dim]--env · --out[/dim]"),
            ("apply", "Execute the contract  [dim]--yes · --dry-run · --provider[/dim]"),
            ("verify", "Confirm deployed state matches the contract"),
        ],
    )

    # ── Generation & Scaffolding ────────────────────────────────────
    _section(
        "▸",
        "Generation & Scaffolding",
        "green",
        [
            ("generate-airflow", "Produce Airflow DAG  [dim](GCP · AWS · Snowflake)[/dim]"),
            ("export", "Export to Airflow · Dagster · Prefect  [dim]--engine[/dim]"),
            ("forge", "AI-powered project creation  [dim]--mode copilot[/dim]"),
            ("blueprint", "Browse / scaffold reusable templates"),
            ("scaffold-ci", "Generate CI/CD pipeline config"),
        ],
    )

    # ── Enterprise Integration & Publishing ─────────────────────────
    _section(
        "▸",
        "Enterprise Integration & Publishing",
        "bright_magenta",
        [
            ("export-opds", "Export to ODPS [dim](Open Data Product Spec)[/dim]"),
            ("odcs", "ODCS v3.1 [dim](Open Data Contract Standard — Bitol.io)[/dim]"),
            ("odps", "ODPS v4.1 [dim](Linux Foundation data product spec)[/dim]"),
            ("market", "Browse & install marketplace data products"),
            ("publish", "Publish a data product to the marketplace"),
        ],
    )

    # ── Quality & Governance ────────────────────────────────────────
    _section(
        "▸",
        "Quality & Governance",
        "yellow",
        [
            ("test", "Test contract against live data  [dim]--output json · --strict[/dim]"),
            ("contract-tests", "Run contract test suites"),
            ("contract-validation", "Deep semantic validation"),
            ("policy-check", "Governance & compliance checks  [dim]--strict[/dim]"),
            ("diff", "Detect drift from deployed state"),
        ],
    )

    # ── Utilities & System ──────────────────────────────────────────
    _section(
        "▸",
        "Utilities & System",
        "bright_white",
        [
            ("doctor", "Check system health & dependencies"),
            ("auth", "Manage cloud provider credentials  [dim]login · status · logout[/dim]"),
            ("admin", "System admin  [dim]diagnostics · test · status · registry[/dim]"),
            ("version", "Show version info"),
            ("viz-graph", "Generate lineage graph  [dim](DOT · PNG)[/dim]"),
        ],
    )

    # ── More commands ───────────────────────────────────────────────
    extras = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    extras.add_column(style="dim bright_cyan", min_width=24, max_width=24)
    extras.add_column(style="dim")
    extras.add_row("  context", "Switch project / environment")
    extras.add_row("  copilot", "Interactive AI assistant")
    extras.add_row("  execute", "Run build jobs manually")
    extras.add_row("  wizard", "Step-by-step guided setup")
    extras.add_row("  marketplace", "Extended marketplace browser")
    extras.add_row("  odps-bitol", "ODPS-Bitol v1.0 (Entropy Data)")
    extras.add_row("  preview", "Dry-run alias for apply")
    extras.add_row("  viz-plan", "Interactive plan visualization")
    console.print("  [dim]▸ More Commands[/dim]")
    console.print(extras)
    console.print()

    # ── Quick-start ─────────────────────────────────────────────────
    console.print(f"  {bar}")
    console.print(
        "  [bold bright_green]⚡ Quick Start[/bold bright_green]     "
        "[bright_cyan]fluid init my-project --quickstart[/bright_cyan]  →  "
        "[bright_cyan]fluid validate contract.fluid.yaml[/bright_cyan]  →  "
        "[bright_cyan]fluid apply contract.fluid.yaml --yes[/bright_cyan]"
    )
    console.print(f"  {bar}")

    # ── Footer ──────────────────────────────────────────────────────
    console.print(
        "  [dim]Docs[/dim]  [bright_cyan]https://github.com/open-data-protocol/fluid[/bright_cyan]   "
        "[dim]│[/dim]   "
        "[dim]Health[/dim]  [bright_cyan]fluid doctor[/bright_cyan]"
    )
    console.print(
        "  [dim italic]Made with ❤️  by DustLabs.co.za — building the future of declarative data engineering[/dim italic]"
    )
    console.print()


def print_forge_help() -> None:
    """Print beautiful help specifically for forge command"""
    if not RICH_AVAILABLE:
        return False

    console = Console()

    # Header
    console.print()
    console.print(
        Panel(
            "[bold bright_magenta]🔨 FLUID Forge[/bold bright_magenta] [dim bright_white]v1.0.0[/dim bright_white]\n"
            "[bright_white]The One Command You Need to Know[/bright_white]\n"
            f"[dim]Create FLUID {_VERSION} data products with AI assistance[/dim]",
            border_style="bright_magenta",
            padding=(1, 2),
            title="[bold bright_white]✨ Project Generator[/bold bright_white]",
            title_align="left",
        )
    )
    console.print()

    # Usage
    console.print("[bold bright_white]Usage:[/bold bright_white]")
    console.print("  [bright_cyan]fluid forge[/bright_cyan] [yellow][OPTIONS][/yellow]")
    console.print(
        "  [bright_cyan]fluid forge[/bright_cyan] [yellow]--mode[/yellow] [bright_green]copilot[/bright_green]"
    )
    console.print(
        "  [bright_cyan]fluid forge[/bright_cyan] [yellow]--template[/yellow] [bright_white]analytics[/bright_white] [yellow]--provider[/yellow] [bright_white]gcp[/bright_white]"
    )
    console.print()

    # Creation Modes
    modes_table = Table(
        show_header=False, box=box.ROUNDED, padding=(0, 2), border_style="bright_magenta"
    )
    modes_table.add_column(style="bright_magenta bold", width=20)
    modes_table.add_column(style="bright_white")

    modes_table.add_row("copilot", "🤖 AI-powered intelligent project creation (recommended)")
    modes_table.add_row("agent", "🎯 Specialized domain experts for specific industries")
    modes_table.add_row("template", "📋 Traditional template-based creation")
    modes_table.add_row("blueprint", "🏗️  Complete enterprise data product templates")

    console.print(
        "[bold bright_magenta]🎨 Creation Modes[/bold bright_magenta] [dim](Choose your workflow)[/dim]"
    )
    console.print(modes_table)
    console.print()

    # Key Options
    options_table = Table(
        show_header=True, box=box.ROUNDED, padding=(0, 1), border_style="bright_yellow"
    )
    options_table.add_column("Option", style="bright_yellow bold", width=30)
    options_table.add_column("Description", style="bright_white")

    options_table.add_row("--mode, -m", "Creation mode (copilot/agent/template/blueprint)")
    options_table.add_row("--agent, -a", "Specific AI agent (finance/healthcare/retail)")
    options_table.add_row("--template, -t", "Project template name")
    options_table.add_row("--provider, -p", "Infrastructure provider (gcp/aws/snowflake/local)")
    options_table.add_row("--blueprint, -b", "Enterprise blueprint name")
    options_table.add_row("--target-dir, -d", "Target directory for project creation")
    options_table.add_row("--quickstart, -q", "Skip confirmations, use recommended defaults")
    options_table.add_row("--interactive, -i", "Force interactive mode")
    options_table.add_row("--dry-run", "Preview without creating files")
    options_table.add_row("--context", "Additional AI context (JSON string or file)")
    options_table.add_row("--llm-provider", "Built-in copilot adapter (openai/anthropic/gemini/ollama)")
    options_table.add_row("--llm-model", "Model identifier for copilot mode")
    options_table.add_row("--llm-endpoint", "Exact HTTP endpoint override for the selected adapter")
    options_table.add_row("--discover / --no-discover", "Enable or disable local metadata discovery")
    options_table.add_row("--discovery-path", "Extra local file or directory to scan for metadata")
    options_table.add_row("--memory / --no-memory", "Enable or disable loading repo-local copilot memory")
    options_table.add_row("--save-memory", "Persist repo-local copilot memory after a successful non-interactive run")
    options_table.add_row("--show-memory", "Show the current project-scoped copilot memory summary and exit")
    options_table.add_row("--reset-memory", "Delete the current project-scoped copilot memory file and exit")

    console.print("[bold bright_yellow]⚙️  Options[/bold bright_yellow]")
    console.print(options_table)
    console.print()

    # Examples
    console.print("[bold bright_green]💡 Quick Start Examples[/bold bright_green]")
    console.print()

    examples = [
        ("AI Copilot Mode (Recommended):", "fluid forge", "Interactive copilot with discovery and validation"),
        (
            "OpenAI Copilot:",
            "fluid forge --mode copilot --llm-provider openai --llm-model gpt-4o-mini",
            "Generate a validated FLUID contract with OpenAI",
        ),
        (
            "Local Ollama:",
            "fluid forge --mode copilot --llm-provider ollama --llm-model llama3.1 --llm-endpoint http://localhost:11434/v1/chat/completions",
            "Use a local model through the built-in Ollama adapter",
        ),
        (
            "Non-Interactive Memory Save:",
            "fluid forge --mode copilot --non-interactive --save-memory",
            "Persist project-scoped copilot memory after a successful run",
        ),
        (
            "Inspect Saved Memory:",
            "fluid forge --show-memory",
            "See what copilot currently remembers about this project",
        ),
        (
            "Reset Saved Memory:",
            "fluid forge --reset-memory",
            "Clear the saved copilot memory for this project",
        ),
        (
            "Specific Template:",
            "fluid forge --template analytics --provider gcp",
            "Use pre-built analytics template",
        ),
        (
            "Domain Expert:",
            "fluid forge --mode agent --agent finance",
            "Finance-specific best practices",
        ),
        (
            "Enterprise Blueprint:",
            "fluid forge --mode blueprint --blueprint customer-360",
            "Complete enterprise solution",
        ),
        ("Quick Start:", "fluid forge --quickstart", "Use smart defaults, no questions"),
        (
            "Preview First:",
            "fluid forge --dry-run --template ml_pipeline",
            "See what will be created",
        ),
    ]

    for desc, cmd, help_text in examples:
        console.print(f"  [bold bright_white]{desc}[/bold bright_white] [dim]{help_text}[/dim]")
        syntax = Syntax(f"  {cmd}", "bash", theme="monokai", padding=(0, 2))
        console.print(syntax)

    # Workflow
    console.print()
    workflow_panel = Panel(
        "[bold]Step 1:[/bold] Run [bright_cyan]fluid forge[/bright_cyan]\n"
        "[bold]Step 2:[/bold] Answer a few questions about your project\n"
        "[bold]Step 3:[/bold] Copilot discovers local metadata and generates a full contract\n"
        "[bold]Step 4:[/bold] Forge validates and repairs the contract if needed\n"
        "[bold]Step 5:[/bold] Forge scaffolds only after validation passes\n"
        "[bold]Step 6:[/bold] Forge shows how memory influenced the run\n"
        "[bold]Step 7:[/bold] Save project-scoped memory only if you explicitly opt in\n\n"
        "[dim]💡 First time? Just run [bright_cyan]fluid forge[/bright_cyan] and follow the prompts![/dim]",
        title="[bold bright_green]🚀 How It Works[/bold bright_green]",
        border_style="bright_green",
        padding=(1, 2),
    )
    console.print(workflow_panel)
    console.print()

    # Tips
    tips_panel = Panel(
        "💡 [bold]Pro Tips:[/bold]\n\n"
        "  • Start with [bright_cyan]--mode copilot[/bright_cyan] for AI-guided creation\n"
        "  • Use [yellow]--dry-run[/yellow] to preview before generating\n"
        "  • Use [yellow]--save-memory[/yellow] for non-interactive runs that should remember project conventions\n"
        "  • Use [yellow]--show-memory[/yellow] to inspect what copilot remembers for this project\n"
        "  • Try [yellow]--quickstart[/yellow] for instant setup with smart defaults\n"
        "  • Explore templates: [bright_cyan]fluid market search[/bright_cyan]\n"
        "  • Get help anytime: [bright_cyan]fluid doctor[/bright_cyan]",
        border_style="bright_yellow",
        padding=(1, 2),
        title="[bold]✨ Tips[/bold]",
        title_align="left",
    )
    console.print(tips_panel)
    console.print()

    # Footer
    console.print(
        Panel(
            "[bold bright_cyan]📚 Learn More[/bold bright_cyan]\n"
            "   https://github.com/open-data-protocol/fluid/docs/forge\n\n"
            "[bold bright_green]💬 Need Help?[/bold bright_green]\n"
            "   Run: [bright_cyan]fluid forge[/bright_cyan] and let AI guide you\n"
            "   Or: [bright_cyan]fluid doctor[/bright_cyan] to check your setup\n\n"
            "[dim]Made with ❤️  for data engineers everywhere[/dim]",
            title="[bold bright_white]📖 Resources[/bold bright_white]",
            title_align="left",
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )

    return True


# ── Enrichment data for bare commands ────────────────────────────────────
# Keyed by command name.  Values are (description, examples_epilog).
# Only needed for commands whose register() doesn't set these on the parser.
_COMMAND_ENRICHMENT: dict[str, tuple[str, str]] = {
    "init": (
        "Create a new data product project — quickstart, wizard, scan, or blank skeleton.",
        "",  # init already has a good argparse help; just add the description
    ),
    "apply": (
        "Execute a FLUID contract end-to-end: provision, transform, govern, deploy.",
        "",  # keep existing epilog
    ),
    "plan": (
        "Generate an execution plan showing every action before you commit.",
        "",  # keep existing epilog
    ),
    "verify": (
        "Confirm that deployed resources match the contract specification.",
        "",  # keep existing epilog
    ),
    "export": (
        "Export a FLUID contract as executable orchestration code (Airflow, Dagster, Prefect).",
        "",
    ),
    "market": (
        "Discover and search data products across enterprise catalogs and marketplaces.",
        "",  # keep existing epilog
    ),
    "generate-airflow": (
        "Generate an Airflow DAG from a FLUID contract. (Deprecated — use fluid export instead.)",
        "",  # keep existing epilog
    ),
    "execute": (
        "Run build jobs defined in a FLUID contract's execution configuration.",
        "",  # keep existing epilog
    ),
    "product-add": (
        "Append a source, exposure, or data quality check to an existing FLUID contract.",
        (
            "  fluid product-add contract.fluid.yaml source --id raw_events --type table\n"
            "  fluid product-add contract.fluid.yaml exposure --id public_api --location /api/v1\n"
            "  fluid product-add contract.fluid.yaml dq --id freshness_check --type freshness"
        ),
    ),
    "odps-bitol": (
        "Work with ODPS-Bitol format for Entropy Data marketplace integration.",
        "",  # keep existing epilog
    ),
    "validate": (
        "Validate a FLUID contract against schemas, provider rules, and best practices.",
        (
            "  fluid validate contract.fluid.yaml\n"
            "  fluid validate contract.fluid.yaml --env prod\n"
            "  fluid validate contract.fluid.yaml --strict --format json"
        ),
    ),
    "scaffold-ci": (
        "Generate a ready-to-use CI/CD pipeline configuration for GitLab or GitHub Actions.",
        (
            "  fluid scaffold-ci contract.fluid.yaml --system github\n"
            "  fluid scaffold-ci contract.fluid.yaml --system gitlab --out .gitlab-ci.yml"
        ),
    ),
    "export-opds": (
        "Export a FLUID contract to OPDS (Open Data Product Specification) JSON.",
        (
            "  fluid export-opds contract.fluid.yaml\n"
            "  fluid export-opds contract.fluid.yaml --out my-product.opds.json\n"
            "  fluid export-opds contract.fluid.yaml --env prod --out prod-product.json"
        ),
    ),
    "contract-tests": (
        "Run schema compatibility and consumer-impact tests against a contract baseline.",
        (
            "  fluid contract-tests contract.fluid.yaml\n"
            "  fluid contract-tests contract.fluid.yaml --baseline schema-v1.json\n"
            "  fluid contract-tests contract.fluid.yaml --env staging"
        ),
    ),
    "test": (
        "Test a contract against live data — schema checks, quality rules, and SLAs.",
        (
            "  fluid test contract.fluid.yaml\n"
            "  fluid test contract.fluid.yaml --output json\n"
            "  fluid test contract.fluid.yaml --strict --server my-account.snowflakecomputing.com\n"
            "  fluid test contract.fluid.yaml --output junit --output-file results.xml"
        ),
    ),
    "preview": (
        "Validate → Plan → Visualize in one step — without applying any changes.",
        (
            "  fluid preview contract.fluid.yaml\n"
            "  fluid preview contract.fluid.yaml --html report.html\n"
            "  fluid preview contract.fluid.yaml --env dev --out plan.json"
        ),
    ),
    "viz-plan": (
        "Render an interactive HTML visualization of an execution plan.",
        (
            "  fluid viz-plan runtime/plan.json\n"
            "  fluid viz-plan runtime/plan.json --out pipeline.html"
        ),
    ),
    "policy-apply": (
        "Apply compiled IAM / access-policy bindings to the target provider.",
        (
            "  fluid policy-apply runtime/policy/bindings.json\n"
            "  fluid policy-apply bindings.json --mode check   # dry-run\n"
            "  fluid policy-apply bindings.json --mode enforce  # live"
        ),
    ),
    "policy-compile": (
        "Compile the contract's accessPolicy section into provider-native IAM bindings.",
        (
            "  fluid policy-compile contract.fluid.yaml\n"
            "  fluid policy-compile contract.fluid.yaml --env prod\n"
            "  fluid policy-compile contract.fluid.yaml --out runtime/policy/bindings.json"
        ),
    ),
    "scaffold-composer": (
        "Generate a Google Cloud Composer DAG from a FLUID contract.",
        (
            "  fluid scaffold-composer contract.fluid.yaml\n"
            "  fluid scaffold-composer contract.fluid.yaml --out-dir dags/\n"
            "  fluid scaffold-composer contract.fluid.yaml --env prod --out-dir dags/"
        ),
    ),
    "product-new": (
        "Bootstrap a new data-product skeleton with folder structure and starter contract.",
        (
            "  fluid product-new --id gold.customer360_v1\n"
            "  fluid product-new --id silver.events_v2 --out-dir ./data-products/"
        ),
    ),
    "docs": (
        "Auto-generate static documentation site from your data product contracts.",
        (
            "  fluid docs\n"
            "  fluid docs --src ./contracts --out ./site\n"
            "  fluid docs --src /data-products --out /var/www/docs"
        ),
    ),
    "providers": (
        "List all discoverable infrastructure providers and their capabilities.",
        ("  fluid providers"),
    ),
    "context": (
        "Get or set default provider, project, and region for the current workspace.",
        (
            "  fluid context list\n"
            "  fluid context set provider gcp\n"
            "  fluid context set project my-gcp-project\n"
            "  fluid context set region us-central1\n"
            "  fluid context get provider"
        ),
    ),
    "auth": (
        "Manage authentication credentials for cloud providers (GCP, AWS, Azure, Snowflake, Databricks).",
        (
            "  fluid auth login --provider gcp\n"
            "  fluid auth status\n"
            "  fluid auth logout --provider aws\n"
            "  fluid auth list"
        ),
    ),
    "blueprint": (
        "Browse, search, and deploy reusable data-product blueprints — complete enterprise templates.",
        (
            "  fluid blueprint list\n"
            "  fluid blueprint search customer-360\n"
            "  fluid blueprint describe customer-360\n"
            "  fluid blueprint create customer-360 --out-dir ./my-project\n"
            "  fluid blueprint validate my-blueprint/"
        ),
    ),
    "copilot": (
        "AI-powered assistant that analyzes your project context and suggests improvements.",
        (
            "  fluid copilot interactive         # start chat session\n"
            "  fluid copilot analyze              # analyze current project\n"
            "  fluid copilot suggest              # suggest next steps"
        ),
    ),
    "marketplace": (
        "Search, browse, and instantiate blueprints from the FLUID marketplace.",
        (
            "  fluid marketplace search analytics\n"
            "  fluid marketplace info customer-360\n"
            "  fluid marketplace instantiate customer-360 --out-dir ./my-project\n"
            "  fluid marketplace categories"
        ),
    ),
    "ide": (
        "IDE integration tools — editor setup, language server, shell completions.",
        (
            "  fluid ide setup                    # configure VS Code / JetBrains\n"
            "  fluid ide completion               # install shell tab-completion\n"
            "  fluid ide lsp start                # start language server\n"
            "  fluid ide validate contract.yaml   # per-file validation"
        ),
    ),
    "datamesh-manager": (
        "Publish data products to Entropy Data / Data Mesh Manager. Supports "
        "data products, data contracts, teams, and the full Entropy Data API.",
        (
            "  fluid datamesh-manager publish contract.fluid.yaml\n"
            "  fluid dmm publish contract.yaml --dry-run\n"
            "  fluid dmm publish contract.yaml --with-contract\n"
            "  fluid dmm list\n"
            "  fluid dmm teams"
        ),
    ),
    "wizard": (
        "Step-by-step guided setup wizard with interactive prompts for creating data products.",
        ("  fluid wizard\n" "  fluid wizard --provider gcp\n" "  fluid wizard --skip-preview"),
    ),
    "diff": (
        "Detect configuration drift by comparing contract (desired state) with actual deployed resources.",
        (
            "  fluid diff contract.fluid.yaml\n"
            "  fluid diff contract.fluid.yaml --env prod\n"
            "  fluid diff contract.fluid.yaml --exit-on-drift   # CI/CD gate\n"
            "  fluid diff contract.fluid.yaml --out drift.json"
        ),
    ),
    "doctor": (
        "Run comprehensive diagnostics — checks FLUID core, providers, schemas, and dependencies.",
        ("  fluid doctor\n" "  fluid doctor --verbose\n" "  fluid doctor --features-only"),
    ),
    "admin": (
        "System administration — diagnostics, tests, templates, pipeline scaffolds, registries.",
        (
            "  fluid admin status\n"
            "  fluid admin diagnostics\n"
            "  fluid admin test\n"
            "  fluid admin test-cli\n"
            "  fluid admin templates\n"
            "  fluid admin registry\n"
            "  fluid admin pipeline"
        ),
    ),
    "version": (
        "Display FLUID CLI version, supported spec versions, and system environment info.",
        (
            "  fluid version\n"
            "  fluid version --verbose\n"
            "  fluid version --format json\n"
            "  fluid version --short"
        ),
    ),
    "generate-pipeline": (
        "Create dynamic DataOps CI/CD pipeline configs for GitHub Actions, GitLab CI, Jenkins, and more.",
        (
            "  fluid generate-pipeline --provider github_actions\n"
            "  fluid generate-pipeline --provider gitlab_ci --complexity enterprise\n"
            "  fluid generate-pipeline --interactive\n"
            "  fluid generate-pipeline --provider jenkins --enable-approvals --enable-security-scan"
        ),
    ),
}


def print_command_help(parser: argparse.ArgumentParser, command_name: str) -> None:
    """Print beautiful Rich-formatted help for any individual command."""
    if not RICH_AVAILABLE:
        parser.print_help()
        return

    # Special handling for forge (has its own bespoke layout)
    if command_name == "forge":
        if print_forge_help():
            return

    console = Console()
    W = console.width or 80
    bar = "[dim bright_cyan]" + "━" * (W - 4) + "[/dim bright_cyan]"

    # ── locate the subparser ─────────────────────────────────────────
    subparser = None
    for action in parser._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            if command_name in action.choices:
                subparser = action.choices[command_name]
                break
    if subparser is None:
        parser.print_help()
        return

    # ── resolve description & epilog (enrichment overrides blanks) ──
    desc = subparser.description or ""
    epilog = subparser.epilog or ""
    enrichment = _COMMAND_ENRICHMENT.get(command_name)
    if enrichment:
        if not desc.strip():
            desc = enrichment[0]
        if not epilog.strip():
            epilog = enrichment[1]
    # Compact multi-line descriptions to a single block
    if desc:
        desc = " ".join(desc.split())
        # If description is very long, use just the first sentence
        if len(desc) > 120:
            first_dot = desc.find(". ")
            if first_dot > 0 and first_dot < 120:
                desc = desc[: first_dot + 1]
            else:
                desc = desc[:117] + "..."

    # ── Header ───────────────────────────────────────────────────────
    console.print()
    console.print(f"  {bar}")
    console.print(f"  [bold bright_cyan]🌊 fluid {command_name}[/bold bright_cyan]")
    if desc:
        console.print(f"  [bright_white]{desc}[/bright_white]")
    console.print(f"  {bar}")
    console.print()

    # ── Usage line ───────────────────────────────────────────────────
    # Build a compact usage from the formatter
    usage_parts = [f"[bright_cyan]fluid {command_name}[/bright_cyan]"]
    has_subcommands = False
    sub_choices: list[tuple[str, str]] = []

    for act in (subparser._subparsers._actions if subparser._subparsers else []):
        if isinstance(act, argparse._SubParsersAction):
            has_subcommands = True
            # Build help map from _choices_actions (stores the help text)
            help_map: dict[str, str] = {}
            for ca in getattr(act, "_choices_actions", []):
                help_map[ca.dest] = ca.help or ""
            for name in act.choices:
                sub_choices.append((name, help_map.get(name, "")))
            usage_parts.append(f"[bright_green]<{'|'.join(act.choices)}>[/bright_green]")
            break

    # Positional args (except subcommands)
    for grp in subparser._action_groups:
        for act in grp._group_actions:
            if isinstance(act, (argparse._HelpAction, argparse._SubParsersAction)):
                continue
            if not act.option_strings:  # positional
                meta = act.metavar or act.dest.upper()
                usage_parts.append(f"[bright_white]{meta}[/bright_white]")

    usage_parts.append("[dim][options][/dim]")
    console.print("  [bold]USAGE[/bold]   " + " ".join(usage_parts))
    console.print()

    # ── Subcommands (if present) ─────────────────────────────────────
    if has_subcommands and sub_choices:
        console.print("  [bold bright_green]▸ Commands[/bold bright_green]")
        tbl = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
        tbl.add_column(style="bright_green bold", min_width=26, max_width=26)
        tbl.add_column(style="bright_white")
        for name, hlp in sub_choices:
            tbl.add_row(f"    {name}", hlp)
        console.print(tbl)
        console.print()

    # ── Arguments & Options ──────────────────────────────────────────
    for group in subparser._action_groups:
        actions = [
            a
            for a in group._group_actions
            if not isinstance(a, (argparse._HelpAction, argparse._SubParsersAction))
        ]
        if not actions:
            continue

        # Section title
        title_raw = (group.title or "").strip()
        low = title_raw.lower()
        if "positional" in low:
            label = "▸ Arguments"
            color = "bright_cyan"
        elif low in ("options", "optional arguments"):
            label = "▸ Options"
            color = "yellow"
        else:
            label = f"▸ {title_raw}"
            color = "bright_yellow"

        console.print(f"  [bold {color}]{label}[/bold {color}]")

        tbl = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
        tbl.add_column(style=f"{color}", min_width=26, max_width=32)
        tbl.add_column(style="bright_white")

        for act in actions:
            # Option / positional string
            if act.option_strings:
                opt = ", ".join(act.option_strings)
                if act.metavar:
                    opt += f" [dim]{act.metavar}[/dim]"
                elif act.type and act.type is not bool:
                    opt += f" [dim]{act.dest.upper()}[/dim]"
            else:
                opt = act.metavar or act.dest

            # Help text
            hlp = act.help or ""
            if act.choices and not isinstance(act, argparse._SubParsersAction):
                # Only append choices when they aren't already in the help text
                if not any(str(c) in hlp for c in list(act.choices)[:2]):
                    hlp += "  [dim](" + " · ".join(str(c) for c in act.choices) + ")[/dim]"
            if act.default not in (None, argparse.SUPPRESS, False):
                # Skip if help already mentions this default
                def_str = str(act.default)
                if f"default: {def_str}" not in hlp and f"default:{def_str}" not in hlp:
                    hlp += f"  [dim](default: {def_str})[/dim]"

            tbl.add_row(f"    {opt}", hlp)

        console.print(tbl)
        console.print()

    # ── Examples ─────────────────────────────────────────────────────
    if epilog and epilog.strip():
        lines = epilog.strip().splitlines()
        formatted: list[str] = []
        in_continuation = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                in_continuation = False
                formatted.append("")
                continue
            # Detect start/continuation of a backslash-continued command
            if stripped.endswith("\\"):
                clean = stripped.rstrip("\\").rstrip()
                formatted.append(f"  [bright_cyan]{clean} \\\\[/bright_cyan]")
                in_continuation = True
            elif in_continuation:
                formatted.append(f"    [bright_cyan]{stripped}[/bright_cyan]")
                if not stripped.endswith("\\"):
                    in_continuation = False
            elif stripped.startswith("#"):
                formatted.append(f"  [dim italic]{stripped}[/dim italic]")
            elif stripped.startswith("fluid ") or stripped.startswith("viz-"):
                formatted.append(f"  [bright_cyan]{stripped}[/bright_cyan]")
            else:
                formatted.append(f"  {stripped}")
        console.print("  [bold bright_magenta]▸ Examples[/bold bright_magenta]")
        console.print("\n".join(formatted))
        console.print()

    # ── Footer ───────────────────────────────────────────────────────
    console.print(
        f"  [dim]Run[/dim] [bright_cyan]fluid --help[/bright_cyan] "
        f"[dim]for all commands  │  [/dim]"
        f"[bright_cyan]fluid {command_name} -h[/bright_cyan] [dim](raw argparse)[/dim]"
    )
    console.print()
