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

"""fluid compile — resolve $ref pointers and emit a single bundled contract.

Usage:
    fluid compile contract.fluid.yaml
    fluid compile contract.fluid.yaml --out contract.bundled.fluid.yaml
    fluid compile contract.fluid.yaml --env prod --out bundled.yaml
    fluid compile contract.fluid.yaml --format json --out bundled.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from ..loader import RefResolutionError, compile_contract, load_with_overlay

COMMAND = "compile"


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        COMMAND,
        help="Resolve $ref pointers and emit a single bundled contract",
        description=(
            "Compile a multi-file FLUID contract into a single document by resolving\n"
            "all $ref pointers. Equivalent to 'swagger-cli bundle' for OpenAPI.\n\n"
            "This is useful for:\n"
            "  - Inspecting the fully-resolved contract before apply/validate\n"
            "  - Archiving a snapshot of all fragments as one document\n"
            "  - Sharing a self-contained contract with other tools\n"
            "  - Debugging $ref resolution issues"
        ),
        epilog=(
            "Examples:\n"
            "  fluid compile contract.fluid.yaml                    # print to stdout\n"
            "  fluid compile contract.fluid.yaml --out bundled.yaml # write to file\n"
            "  fluid compile contract.fluid.yaml --env prod         # with overlay\n"
            "  fluid compile contract.fluid.yaml --format json      # JSON output\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("contract", help="Path to the root FLUID contract file")
    p.add_argument(
        "--out",
        "-o",
        default="-",
        help="Output path (default: '-' for stdout)",
    )
    p.add_argument(
        "--env",
        "-e",
        default=None,
        help="Environment overlay to apply after ref resolution",
    )
    p.add_argument(
        "--format",
        "-f",
        choices=["yaml", "json"],
        default=None,
        help="Output format (default: infer from --out extension, else YAML)",
    )
    p.set_defaults(cmd=COMMAND, func=run)


def _infer_format(out: str, explicit: str | None) -> str:
    """Determine output format from --format flag, --out extension, or default to YAML."""
    if explicit:
        return explicit
    if out and out != "-":
        suffix = Path(out).suffix.lower()
        if suffix == ".json":
            return "json"
    return "yaml"


def _serialize(contract: Dict[str, Any], fmt: str) -> str:
    """Serialize a contract dict to YAML or JSON string."""
    if fmt == "json":
        return json.dumps(contract, indent=2, default=str) + "\n"
    if yaml is None:
        raise RuntimeError(
            "YAML output requires PyYAML. Install with: pip install pyyaml\n" "Or use --format json"
        )
    return yaml.dump(
        contract,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )


def run(args: argparse.Namespace, logger: logging.Logger) -> int:
    contract_path = args.contract
    out = args.out
    env = args.env
    fmt = _infer_format(out, args.format)

    try:
        # Compile: resolve all $ref pointers
        compiled = compile_contract(contract_path, logger=logger)

        # Apply environment overlay on top (if requested)
        if env:
            from ..loader import _deep_merge, _overlay_candidates, _parse_file

            base_path = Path(contract_path)
            for cand in _overlay_candidates(base_path, env):
                if cand.exists():
                    overlay = _parse_file(cand)
                    compiled = _deep_merge(dict(compiled), overlay)
                    logger.info("overlay_applied", extra={"overlay": str(cand)})
                    break

    except FileNotFoundError as e:
        sys.stderr.write(f"❌ File not found: {e}\n")
        return 2
    except RefResolutionError as e:
        sys.stderr.write(f"❌ $ref resolution error: {e}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"❌ Compilation failed: {e}\n")
        return 1

    # Serialize
    output = _serialize(compiled, fmt)

    # Write
    if out == "-":
        sys.stdout.write(output)
    else:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(output, encoding="utf-8")
        logger.info("compile_written", extra={"out": str(p), "format": fmt})
        sys.stderr.write(f"✅ Compiled contract written to {p}\n")

    return 0
