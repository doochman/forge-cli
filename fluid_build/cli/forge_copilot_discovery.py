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

"""Local workspace discovery for the forge copilot."""

from __future__ import annotations

__all__ = [
    "DiscoveryReport",
    "discover_local_context",
]

import json
import logging
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from fluid_build.cli.forge_copilot_llm_providers import CopilotGenerationError
from fluid_build.cli.forge_copilot_schema_inference import (
    extract_provider_hints,
    summarize_sample_file,
)
from fluid_build.config import RUN_STATE_DIR
from fluid_build.util.contract import get_builds

LOG = logging.getLogger("fluid.cli.forge_copilot.discovery")

MAX_DISCOVERY_FILES = 300
MAX_SQL_FILES = 25
MAX_READMES = 10
MAX_SAMPLE_FILES = 12
MAX_EXISTING_CONTRACTS = 12
MAX_README_LINES = 80
DISCOVERABLE_SAMPLE_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet", ".pq", ".avro"}
RUN_STATE_PATH_PARTS = tuple(Path(RUN_STATE_DIR).parts)

IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryReport:
    """Metadata-only view of locally discovered assets."""

    workspace_roots: List[str]
    files_scanned: int = 0
    detected_sources: List[Dict[str, Any]] = field(default_factory=list)
    sql_files: List[Dict[str, Any]] = field(default_factory=list)
    dbt_projects: List[Dict[str, Any]] = field(default_factory=list)
    terraform_projects: List[Dict[str, Any]] = field(default_factory=list)
    readmes: List[Dict[str, Any]] = field(default_factory=list)
    existing_contracts: List[Dict[str, Any]] = field(default_factory=list)
    sample_files: List[Dict[str, Any]] = field(default_factory=list)
    provider_hints: List[str] = field(default_factory=list)
    build_constraints: List[str] = field(default_factory=list)
    discovery_warnings: List[str] = field(default_factory=list)

    def to_prompt_payload(self) -> Dict[str, Any]:
        """Return a metadata-only payload safe to share with the LLM."""
        home = str(Path.home())
        sanitized_roots = [
            root.replace(home, "~") if isinstance(root, str) else root
            for root in self.workspace_roots
        ]
        return {
            "workspace_roots": sanitized_roots,
            "files_scanned": self.files_scanned,
            "detected_sources": self.detected_sources,
            "sql_files": self.sql_files,
            "dbt_projects": self.dbt_projects,
            "terraform_projects": self.terraform_projects,
            "readmes": self.readmes,
            "existing_contracts": self.existing_contracts,
            "sample_files": self.sample_files,
            "provider_hints": self.provider_hints,
            "build_constraints": self.build_constraints,
            "discovery_warnings": self.discovery_warnings,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_local_context(
    discovery_path: Optional[str],
    *,
    discover: bool = True,
    workspace_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> DiscoveryReport:
    """Scan local workspace files and return a metadata-only discovery report."""
    root = (workspace_root or Path.cwd()).resolve()
    roots = [root]
    if discovery_path:
        extra = Path(discovery_path).expanduser().resolve()
        if not extra.exists():
            raise CopilotGenerationError(
                "copilot_discovery_path_missing",
                f"Discovery path does not exist: {extra}",
                suggestions=["Check the path passed to --discovery-path"],
            )
        if extra not in roots:
            roots.append(extra)

    report = DiscoveryReport(
        workspace_roots=[str(path) for path in roots],
        build_constraints=[
            "Discovery payload must exclude raw sample rows, full file contents, and credentials.",
            "Use only providers and templates supported by the local Forge registries.",
            "Prefer placeholder env vars for destination configuration instead of hard-coded secrets.",
        ],
    )

    if not discover:
        report.build_constraints.append("Discovery was disabled by the user.")
        return report

    seen_files: set[Path] = set()
    provider_counts: Counter[str] = Counter()
    detected_sources: List[Dict[str, Any]] = []

    for scan_root in roots:
        for path in _iter_candidate_files(scan_root):
            if path in seen_files:
                continue
            if _is_excluded_discovery_artifact(path):
                continue
            seen_files.add(path)
            report.files_scanned += 1
            suffix = path.suffix.lower()

            if path.name == "dbt_project.yml":
                project = _summarize_dbt_project(path)
                report.dbt_projects.append(project)
                provider_counts.update(project.get("provider_hints") or [])
                continue

            if suffix == ".tf":
                terraform = _summarize_terraform_file(path)
                report.terraform_projects.append(terraform)
                provider_counts.update(terraform.get("provider_hints") or [])
                continue

            if path.name.lower().startswith("readme") and len(report.readmes) < MAX_READMES:
                report.readmes.append(_summarize_readme(path))
                continue

            if path.name.endswith("contract.fluid.yaml") or path.name.endswith(
                "contract.fluid.json"
            ):
                if len(report.existing_contracts) < MAX_EXISTING_CONTRACTS:
                    summary = _summarize_existing_contract(path)
                    report.existing_contracts.append(summary)
                    provider_counts.update(summary.get("providers") or [])
                continue

            if suffix == ".sql" and len(report.sql_files) < MAX_SQL_FILES:
                report.sql_files.append(_summarize_sql_file(path))
                continue

            if (
                suffix in DISCOVERABLE_SAMPLE_SUFFIXES
                and len(report.sample_files) < MAX_SAMPLE_FILES
            ):
                sample = summarize_sample_file(path)
                report.sample_files.append(sample)
                detected_sources.append(sample)
                provider_counts.update(sample.get("provider_hints") or [])
                report.discovery_warnings.extend(sample.get("warnings") or [])

            if report.files_scanned >= MAX_DISCOVERY_FILES:
                break
        if report.files_scanned >= MAX_DISCOVERY_FILES:
            break

    report.detected_sources = detected_sources[:MAX_SAMPLE_FILES]
    report.provider_hints = [name for name, _ in provider_counts.most_common()]

    if report.sql_files:
        report.build_constraints.append(
            "Existing SQL assets were found; reuse discovered source table names and output naming conventions where possible."
        )
    if any(sample.get("format") == "parquet" for sample in report.sample_files):
        report.build_constraints.append(
            "Parquet files were discovered; prefer discovered column names, logical types, and storage conventions instead of inventing schemas."
        )
    if any(sample.get("format") == "avro" for sample in report.sample_files):
        report.build_constraints.append(
            "Avro files were discovered; preserve the discovered record shape and union/logical-type intent when generating exposes and builds."
        )
    if report.existing_contracts:
        report.build_constraints.append(
            "Existing FLUID contracts were found; stay consistent with discovered contract naming and provider conventions."
        )

    if logger:
        logger.debug(
            "copilot_discovery_complete",
            extra={
                "files_scanned": report.files_scanned,
                "provider_hints": report.provider_hints,
            },
        )

    return report


# ---------------------------------------------------------------------------
# File traversal
# ---------------------------------------------------------------------------


def _iter_candidate_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return

    queue: deque[Path] = deque([root])
    yielded = 0
    while queue and yielded < MAX_DISCOVERY_FILES:
        current = queue.popleft()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name in IGNORED_DIRECTORIES:
                continue
            if entry.is_dir():
                queue.append(entry)
                continue
            yielded += 1
            yield entry
            if yielded >= MAX_DISCOVERY_FILES:
                return


def _is_run_state_artifact(path: Path) -> bool:
    parts = path.parts
    if len(parts) < len(RUN_STATE_PATH_PARTS):
        return False
    for index in range(len(parts) - len(RUN_STATE_PATH_PARTS) + 1):
        if tuple(parts[index : index + len(RUN_STATE_PATH_PARTS)]) == RUN_STATE_PATH_PARTS:
            return True
    return False


def _is_excluded_discovery_artifact(path: Path) -> bool:
    if _is_run_state_artifact(path):
        return True
    return any(part.lower() == "airbyte" for part in path.parts)


# ---------------------------------------------------------------------------
# File summarizers
# ---------------------------------------------------------------------------


def _summarize_dbt_project(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {"path": str(path), "warnings": [f"Could not parse dbt project file: {path.name}"]}
    profile = data.get("profile")
    return {
        "path": str(path),
        "name": data.get("name"),
        "profile": profile,
        "model_paths": data.get("model-paths") or [],
        "provider_hints": extract_provider_hints(" ".join([str(profile), str(data)])),
    }


def _summarize_terraform_file(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    resource_matches = re.findall(r'resource\s+"([^"]+)"\s+"([^"]+)"', content)
    return {
        "path": str(path),
        "resources": [
            {"type": resource_type, "name": name} for resource_type, name in resource_matches[:15]
        ],
        "provider_hints": extract_provider_hints(content),
    }


def _summarize_readme(path: Path) -> Dict[str, Any]:
    headings: List[str] = []
    words = 0
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if index >= MAX_README_LINES:
            break
        if line.strip().startswith("#"):
            headings.append(line.lstrip("#").strip())
        words += len(line.split())
    return {"path": str(path), "headings": headings[:12], "word_count": words}


def _summarize_existing_contract(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            contract = json.loads(raw)
        else:
            contract = yaml.safe_load(raw)
    except Exception:  # noqa: BLE001
        return {"path": str(path), "warnings": [f"Could not parse contract file: {path.name}"]}
    if not isinstance(contract, dict):
        return {
            "path": str(path),
            "warnings": [f"Contract file is not a valid mapping: {path.name}"],
        }
    providers = []
    for expose in contract.get("exposes") or []:
        binding = expose.get("binding") or {}
        if binding.get("platform"):
            providers.append(binding["platform"])
    return {
        "path": str(path),
        "fluid_version": contract.get("fluidVersion"),
        "kind": contract.get("kind"),
        "id": contract.get("id"),
        "name": contract.get("name"),
        "providers": sorted(set(providers)),
        "build_ids": [build.get("id") for build in get_builds(contract)[:10] if build.get("id")],
        "expose_ids": [
            expose.get("exposeId")
            for expose in (contract.get("exposes") or [])[:10]
            if expose.get("exposeId")
        ],
    }


def _summarize_sql_file(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    table_refs = re.findall(
        r"\b(?:from|join|into|update|table)\s+([A-Za-z0-9_.`\"]+)",
        content,
        flags=re.IGNORECASE,
    )
    return {
        "path": str(path),
        "line_count": len(content.splitlines()),
        "referenced_tables": table_refs[:15],
    }
