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

"""Tests for $ref resolution and multi-file contract compilation.

Covers:
  - Basic external file $ref resolution
  - Nested / transitive $ref resolution
  - $ref inside lists (builds, exposes arrays)
  - JSON pointer fragments (file.yaml#/section)
  - Circular reference detection
  - Missing file errors
  - Depth limit protection
  - JSON and YAML format support
  - compile_contract() entry point
  - load_contract() transparent resolution
  - load_with_overlay() with refs + overlay
  - Backward compatibility (contracts without $ref unchanged)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from fluid_build.loader import (
    RefResolutionError,
    _is_ref_node,
    _parse_ref,
    _resolve_pointer,
    _resolve_refs,
    compile_contract,
    load_contract,
    load_with_overlay,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_yaml(path: Path, data: dict | list | str) -> Path:
    """Write a YAML file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
    return path


def _write_json_file(path: Path, data: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ===========================================================================
# Unit tests: helper functions
# ===========================================================================
class TestIsRefNode:
    def test_valid_ref(self):
        assert _is_ref_node({"$ref": "./file.yaml"}) is True

    def test_ref_with_extra_keys_is_not_ref(self):
        assert _is_ref_node({"$ref": "./file.yaml", "extra": True}) is False

    def test_empty_dict(self):
        assert _is_ref_node({}) is False

    def test_non_dict(self):
        assert _is_ref_node("string") is False
        assert _is_ref_node([1, 2]) is False
        assert _is_ref_node(None) is False


class TestParseRef:
    def test_simple_file(self):
        assert _parse_ref("./file.yaml") == ("./file.yaml", None)

    def test_file_with_pointer(self):
        assert _parse_ref("./file.yaml#/User") == ("./file.yaml", "/User")

    def test_same_file_pointer(self):
        assert _parse_ref("#/definitions/common") == ("", "/definitions/common")

    def test_no_pointer_after_hash(self):
        assert _parse_ref("./file.yaml#") == ("./file.yaml", None)


class TestResolvePointer:
    def test_root_pointer(self):
        obj = {"a": 1}
        assert _resolve_pointer(obj, "/") is obj

    def test_empty_pointer(self):
        obj = {"a": 1}
        assert _resolve_pointer(obj, "") is obj

    def test_single_key(self):
        assert _resolve_pointer({"builds": [1, 2]}, "/builds") == [1, 2]

    def test_nested_key(self):
        obj = {"a": {"b": {"c": 42}}}
        assert _resolve_pointer(obj, "/a/b/c") == 42

    def test_list_index(self):
        obj = {"items": ["zero", "one", "two"]}
        assert _resolve_pointer(obj, "/items/1") == "one"

    def test_missing_key_raises(self):
        with pytest.raises(RefResolutionError, match="not found"):
            _resolve_pointer({"a": 1}, "/missing")

    def test_invalid_list_index_raises(self):
        with pytest.raises(RefResolutionError, match="not a valid list index"):
            _resolve_pointer({"items": [1, 2]}, "/items/abc")


# ===========================================================================
# Integration tests: _resolve_refs
# ===========================================================================
class TestResolveRefsBasic:
    """Basic single-level $ref resolution."""

    def test_simple_external_ref(self, tmp_path):
        _write_yaml(tmp_path / "policy.yaml", {"classification": "Internal", "authn": "iam"})

        tree = {"name": "test", "policy": {"$ref": "./policy.yaml"}}
        result = _resolve_refs(tree, tmp_path)

        assert result["name"] == "test"
        assert result["policy"] == {"classification": "Internal", "authn": "iam"}

    def test_ref_in_list(self, tmp_path):
        _write_yaml(tmp_path / "build.yaml", {"id": "ingest", "engine": "python"})

        tree = {"builds": [{"$ref": "./build.yaml"}]}
        result = _resolve_refs(tree, tmp_path)

        assert result["builds"] == [{"id": "ingest", "engine": "python"}]

    def test_multiple_refs_in_list(self, tmp_path):
        _write_yaml(tmp_path / "expose1.yaml", {"exposeId": "table1", "kind": "table"})
        _write_yaml(tmp_path / "expose2.yaml", {"exposeId": "table2", "kind": "view"})

        tree = {"exposes": [{"$ref": "./expose1.yaml"}, {"$ref": "./expose2.yaml"}]}
        result = _resolve_refs(tree, tmp_path)

        assert len(result["exposes"]) == 2
        assert result["exposes"][0]["exposeId"] == "table1"
        assert result["exposes"][1]["exposeId"] == "table2"

    def test_no_refs_passthrough(self, tmp_path):
        tree = {"name": "test", "builds": [{"id": "b1"}], "scalar": 42}
        result = _resolve_refs(tree, tmp_path)
        assert result == tree

    def test_scalar_passthrough(self, tmp_path):
        assert _resolve_refs("hello", tmp_path) == "hello"
        assert _resolve_refs(42, tmp_path) == 42
        assert _resolve_refs(None, tmp_path) is None


class TestResolveRefsNested:
    """Transitive / nested $ref resolution."""

    def test_transitive_refs(self, tmp_path):
        """A refs B which refs C — all should resolve."""
        _write_yaml(tmp_path / "c.yaml", {"value": "deep"})
        _write_yaml(tmp_path / "b.yaml", {"inner": {"$ref": "./c.yaml"}})

        tree = {"top": {"$ref": "./b.yaml"}}
        result = _resolve_refs(tree, tmp_path)

        assert result["top"]["inner"]["value"] == "deep"

    def test_ref_in_subdirectory(self, tmp_path):
        builds_dir = tmp_path / "builds"
        builds_dir.mkdir()
        _write_yaml(builds_dir / "ingest.yaml", {"id": "ingest", "engine": "python"})

        tree = {"builds": [{"$ref": "./builds/ingest.yaml"}]}
        result = _resolve_refs(tree, tmp_path)

        assert result["builds"][0]["id"] == "ingest"

    def test_ref_relative_to_fragment_dir(self, tmp_path):
        """When B.yaml refs C.yaml, C is resolved relative to B's directory."""
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_yaml(sub / "c.yaml", {"val": "resolved"})
        _write_yaml(sub / "b.yaml", {"inner": {"$ref": "./c.yaml"}})

        tree = {"top": {"$ref": "./sub/b.yaml"}}
        result = _resolve_refs(tree, tmp_path)

        assert result["top"]["inner"]["val"] == "resolved"


class TestResolveRefsPointer:
    """$ref with JSON pointer fragment."""

    def test_file_with_pointer(self, tmp_path):
        _write_yaml(
            tmp_path / "shared.yaml",
            {"sovereignty": {"jurisdiction": "EU"}, "other": "ignored"},
        )

        tree = {"sov": {"$ref": "./shared.yaml#/sovereignty"}}
        result = _resolve_refs(tree, tmp_path)

        assert result["sov"] == {"jurisdiction": "EU"}

    def test_pointer_into_list(self, tmp_path):
        _write_yaml(tmp_path / "data.yaml", {"items": ["a", "b", "c"]})

        tree = {"val": {"$ref": "./data.yaml#/items/1"}}
        result = _resolve_refs(tree, tmp_path)

        assert result["val"] == "b"


class TestResolveRefsJSON:
    """$ref to JSON files."""

    def test_ref_to_json_file(self, tmp_path):
        _write_json_file(
            tmp_path / "policy.json",
            {"classification": "Restricted", "authn": "oauth2"},
        )

        tree = {"policy": {"$ref": "./policy.json"}}
        result = _resolve_refs(tree, tmp_path)

        assert result["policy"]["classification"] == "Restricted"


class TestResolveRefsDiamond:
    """Diamond dependency — same file referenced from multiple branches."""

    def test_same_file_from_two_branches(self, tmp_path):
        """Two sibling keys ref the same file — should NOT raise circular."""
        _write_yaml(tmp_path / "shared.yaml", {"value": "shared"})

        tree = {
            "branch_a": {"$ref": "./shared.yaml"},
            "branch_b": {"$ref": "./shared.yaml"},
        }
        result = _resolve_refs(tree, tmp_path)

        assert result["branch_a"]["value"] == "shared"
        assert result["branch_b"]["value"] == "shared"

    def test_same_file_different_pointers(self, tmp_path):
        """Two refs to the same file with different JSON pointers."""
        _write_yaml(
            tmp_path / "shared.yaml",
            {"sovereignty": {"jurisdiction": "EU"}, "compliance": {"framework": "GDPR"}},
        )

        tree = {
            "sov": {"$ref": "./shared.yaml#/sovereignty"},
            "comp": {"$ref": "./shared.yaml#/compliance"},
        }
        result = _resolve_refs(tree, tmp_path)

        assert result["sov"]["jurisdiction"] == "EU"
        assert result["comp"]["framework"] == "GDPR"

    def test_diamond_via_transitive_refs(self, tmp_path):
        """A → B → D, A → C → D  (diamond) should resolve without error."""
        _write_yaml(tmp_path / "d.yaml", {"leaf": "value"})
        _write_yaml(tmp_path / "b.yaml", {"inner": {"$ref": "./d.yaml"}})
        _write_yaml(tmp_path / "c.yaml", {"inner": {"$ref": "./d.yaml"}})

        tree = {
            "from_b": {"$ref": "./b.yaml"},
            "from_c": {"$ref": "./c.yaml"},
        }
        result = _resolve_refs(tree, tmp_path)

        assert result["from_b"]["inner"]["leaf"] == "value"
        assert result["from_c"]["inner"]["leaf"] == "value"

    def test_same_file_in_list_twice(self, tmp_path):
        """The same fragment referenced twice in a list."""
        _write_yaml(tmp_path / "build.yaml", {"id": "ingest", "engine": "python"})

        tree = {"builds": [{"$ref": "./build.yaml"}, {"$ref": "./build.yaml"}]}
        result = _resolve_refs(tree, tmp_path)

        assert len(result["builds"]) == 2
        assert result["builds"][0] == result["builds"][1]


class TestResolveRefsDepthLimit:
    """Depth limit protection."""

    def test_deep_nesting_raises(self, tmp_path):
        """Chain of refs deeper than _MAX_REF_DEPTH triggers error."""
        from fluid_build.loader import _MAX_REF_DEPTH

        # Create a chain: file_0 → file_1 → ... → file_(MAX+1)
        for i in range(_MAX_REF_DEPTH + 2):
            if i == _MAX_REF_DEPTH + 1:
                _write_yaml(tmp_path / f"file_{i}.yaml", {"leaf": "end"})
            else:
                _write_yaml(tmp_path / f"file_{i}.yaml", {"next": {"$ref": f"./file_{i+1}.yaml"}})

        tree = {"start": {"$ref": "./file_0.yaml"}}
        with pytest.raises(RefResolutionError, match="depth exceeded"):
            _resolve_refs(tree, tmp_path)


class TestResolveRefsPathVariants:
    """Path variation edge cases."""

    def test_parent_directory_traversal(self, tmp_path):
        """Ref with ../ to a sibling directory."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        _write_yaml(dir_b / "shared.yaml", {"data": "from_sibling"})
        _write_yaml(dir_a / "contract.yaml", {"section": {"$ref": "../b/shared.yaml"}})

        result = load_contract(dir_a / "contract.yaml")
        assert result["section"]["data"] == "from_sibling"

    def test_yml_extension(self, tmp_path):
        """Fragments with .yml extension should work."""
        _write_yaml(tmp_path / "policy.yml", {"classification": "Confidential"})

        tree = {"policy": {"$ref": "./policy.yml"}}
        result = _resolve_refs(tree, tmp_path)

        assert result["policy"]["classification"] == "Confidential"


class TestResolveRefsErrors:
    """Error detection and handling."""

    def test_missing_file_raises(self, tmp_path):
        tree = {"policy": {"$ref": "./nonexistent.yaml"}}
        with pytest.raises(RefResolutionError, match="not found"):
            _resolve_refs(tree, tmp_path)

    def test_circular_ref_raises(self, tmp_path):
        # a.yaml refs b.yaml, b.yaml refs a.yaml
        _write_yaml(tmp_path / "a.yaml", {"inner": {"$ref": "./b.yaml"}})
        _write_yaml(tmp_path / "b.yaml", {"inner": {"$ref": "./a.yaml"}})

        tree = {"top": {"$ref": "./a.yaml"}}
        with pytest.raises(RefResolutionError, match="Circular"):
            _resolve_refs(tree, tmp_path)

    def test_self_referencing_file_raises(self, tmp_path):
        _write_yaml(tmp_path / "self.yaml", {"inner": {"$ref": "./self.yaml"}})

        tree = {"top": {"$ref": "./self.yaml"}}
        with pytest.raises(RefResolutionError, match="Circular"):
            _resolve_refs(tree, tmp_path)

    def test_non_string_ref_raises(self, tmp_path):
        tree = {"policy": {"$ref": 123}}
        with pytest.raises(RefResolutionError, match="must be a string"):
            _resolve_refs(tree, tmp_path)

    def test_invalid_pointer_raises(self, tmp_path):
        _write_yaml(tmp_path / "data.yaml", {"a": 1})

        tree = {"val": {"$ref": "./data.yaml#/missing_key"}}
        with pytest.raises(RefResolutionError, match="not found"):
            _resolve_refs(tree, tmp_path)

    def test_same_file_pointer_skipped(self, tmp_path):
        """Same-file refs (#/...) are preserved as-is (future feature)."""
        tree = {"val": {"$ref": "#/definitions/common"}}
        result = _resolve_refs(tree, tmp_path)
        assert result["val"] == {"$ref": "#/definitions/common"}


# ===========================================================================
# Integration tests: compile_contract
# ===========================================================================
class TestCompileContract:
    def test_compile_simple(self, tmp_path):
        _write_yaml(tmp_path / "policy.yaml", {"classification": "Internal"})
        _write_yaml(
            tmp_path / "contract.fluid.yaml",
            {
                "fluidVersion": "0.7.1",
                "kind": "DataProduct",
                "id": "test.product",
                "name": "Test",
                "policy": {"$ref": "./policy.yaml"},
            },
        )

        result = compile_contract(tmp_path / "contract.fluid.yaml")

        assert result["fluidVersion"] == "0.7.1"
        assert result["policy"]["classification"] == "Internal"
        assert "$ref" not in str(result)

    def test_compile_no_refs(self, tmp_path):
        contract = {"fluidVersion": "0.7.1", "kind": "DataProduct", "id": "test"}
        _write_yaml(tmp_path / "contract.fluid.yaml", contract)

        result = compile_contract(tmp_path / "contract.fluid.yaml")
        assert result == contract

    def test_compile_resolve_refs_false(self, tmp_path):
        _write_yaml(
            tmp_path / "contract.fluid.yaml",
            {"policy": {"$ref": "./missing.yaml"}},
        )
        # With resolve_refs=False, should NOT raise even if target is missing
        result = compile_contract(tmp_path / "contract.fluid.yaml", resolve_refs=False)
        assert result["policy"]["$ref"] == "./missing.yaml"

    def test_compile_realistic_multifile(self, tmp_path):
        """Full realistic multi-file contract."""
        _write_yaml(
            tmp_path / "sovereignty.yaml",
            {"jurisdiction": "EU", "allowedRegions": ["europe-west3"]},
        )
        _write_yaml(
            tmp_path / "access.yaml",
            {"grants": [{"principal": "group:analytics@co.com", "permissions": ["read"]}]},
        )
        builds_dir = tmp_path / "builds"
        builds_dir.mkdir()
        _write_yaml(
            builds_dir / "ingest.yaml",
            {"id": "ingest", "engine": "python", "repository": "./runtime"},
        )
        exposes_dir = tmp_path / "exposes"
        exposes_dir.mkdir()
        _write_yaml(
            exposes_dir / "bq_table.yaml",
            {
                "exposeId": "prices_table",
                "kind": "table",
                "binding": {"platform": "gcp", "location": {"project": "p", "dataset": "d"}},
            },
        )
        _write_yaml(
            exposes_dir / "api.yaml",
            {"exposeId": "prices_api", "kind": "api", "binding": {"platform": "gcp"}},
        )

        _write_yaml(
            tmp_path / "contract.fluid.yaml",
            {
                "fluidVersion": "0.7.1",
                "kind": "DataProduct",
                "id": "finance.btc_prices",
                "name": "BTC Prices",
                "sovereignty": {"$ref": "./sovereignty.yaml"},
                "accessPolicy": {"$ref": "./access.yaml"},
                "builds": [{"$ref": "./builds/ingest.yaml"}],
                "exposes": [
                    {"$ref": "./exposes/bq_table.yaml"},
                    {"$ref": "./exposes/api.yaml"},
                ],
            },
        )

        result = compile_contract(tmp_path / "contract.fluid.yaml")

        # All refs resolved
        assert result["sovereignty"]["jurisdiction"] == "EU"
        assert result["accessPolicy"]["grants"][0]["principal"] == "group:analytics@co.com"
        assert result["builds"][0]["id"] == "ingest"
        assert len(result["exposes"]) == 2
        assert result["exposes"][0]["exposeId"] == "prices_table"
        assert result["exposes"][1]["exposeId"] == "prices_api"
        # No $ref remnants
        assert "$ref" not in json.dumps(result)


# ===========================================================================
# Integration tests: load_contract with transparent ref resolution
# ===========================================================================
class TestLoadContractWithRefs:
    def test_transparent_resolution(self, tmp_path):
        _write_yaml(tmp_path / "fragment.yaml", {"data": "resolved"})
        _write_yaml(
            tmp_path / "contract.yaml",
            {"name": "test", "section": {"$ref": "./fragment.yaml"}},
        )

        result = load_contract(tmp_path / "contract.yaml")
        assert result["section"]["data"] == "resolved"

    def test_opt_out_of_resolution(self, tmp_path):
        _write_yaml(
            tmp_path / "contract.yaml",
            {"name": "test", "section": {"$ref": "./fragment.yaml"}},
        )

        result = load_contract(tmp_path / "contract.yaml", resolve_refs=False)
        assert result["section"]["$ref"] == "./fragment.yaml"


# ===========================================================================
# Integration tests: load_with_overlay + refs
# ===========================================================================
class TestLoadWithOverlayAndRefs:
    def test_refs_resolved_before_overlay(self, tmp_path):
        """Refs resolve first, then overlay can override ref'd content."""
        _write_yaml(
            tmp_path / "binding.yaml",
            {"platform": "gcp", "location": {"project": "dev-proj", "region": "us-central1"}},
        )
        _write_yaml(
            tmp_path / "contract.fluid.yaml",
            {
                "fluidVersion": "0.7.1",
                "kind": "DataProduct",
                "id": "test",
                "name": "test",
                "binding": {"$ref": "./binding.yaml"},
            },
        )
        # Overlay changes the project
        overlays = tmp_path / "overlays"
        overlays.mkdir()
        _write_yaml(
            overlays / "prod.yaml",
            {"binding": {"location": {"project": "prod-proj"}}},
        )

        result = load_with_overlay(tmp_path / "contract.fluid.yaml", env="prod")

        # Ref resolved + overlay applied
        assert result["binding"]["platform"] == "gcp"
        assert result["binding"]["location"]["project"] == "prod-proj"


# ===========================================================================
# Backward compatibility
# ===========================================================================
class TestBackwardCompatibility:
    def test_contract_without_refs_unchanged(self, tmp_path):
        contract = {
            "fluidVersion": "0.7.1",
            "kind": "DataProduct",
            "id": "test.product",
            "name": "Test Product",
            "metadata": {"layer": "Gold", "owner": {"team": "de"}},
            "builds": [{"id": "b1", "engine": "python"}],
            "exposes": [{"exposeId": "t1", "kind": "table"}],
        }
        _write_yaml(tmp_path / "contract.fluid.yaml", contract)

        result = load_contract(tmp_path / "contract.fluid.yaml")
        assert result == contract

    def test_load_with_overlay_without_refs(self, tmp_path):
        contract = {"fluidVersion": "0.7.1", "kind": "DataProduct", "id": "test", "name": "t"}
        _write_yaml(tmp_path / "contract.fluid.yaml", contract)

        result = load_with_overlay(tmp_path / "contract.fluid.yaml")
        assert result == contract


# ===========================================================================
# Unit tests: compile CLI helpers
# ===========================================================================
class TestCompileCLIHelpers:
    """Tests for compile.py helper functions."""

    def test_infer_format_explicit_json(self):
        from fluid_build.cli.compile import _infer_format

        assert _infer_format("out.yaml", "json") == "json"

    def test_infer_format_explicit_yaml(self):
        from fluid_build.cli.compile import _infer_format

        assert _infer_format("out.json", "yaml") == "yaml"

    def test_infer_format_from_json_extension(self):
        from fluid_build.cli.compile import _infer_format

        assert _infer_format("out.json", None) == "json"

    def test_infer_format_from_yaml_extension(self):
        from fluid_build.cli.compile import _infer_format

        assert _infer_format("out.yaml", None) == "yaml"

    def test_infer_format_stdout_defaults_yaml(self):
        from fluid_build.cli.compile import _infer_format

        assert _infer_format("-", None) == "yaml"

    def test_serialize_json(self):
        from fluid_build.cli.compile import _serialize

        result = _serialize({"a": 1, "b": [2, 3]}, "json")
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": [2, 3]}

    def test_serialize_yaml(self):
        from fluid_build.cli.compile import _serialize

        result = _serialize({"a": 1, "b": [2, 3]}, "yaml")
        parsed = yaml.safe_load(result)
        assert parsed == {"a": 1, "b": [2, 3]}


class TestCompileCLIRun:
    """Integration tests for the compile command run() function."""

    def test_compile_to_file(self, tmp_path):
        from fluid_build.cli.compile import run

        _write_yaml(tmp_path / "frag.yaml", {"resolved": True})
        _write_yaml(
            tmp_path / "contract.yaml",
            {"name": "test", "section": {"$ref": "./frag.yaml"}},
        )
        out_path = tmp_path / "bundled.yaml"

        args = argparse.Namespace(
            contract=str(tmp_path / "contract.yaml"),
            out=str(out_path),
            env=None,
            format=None,
        )
        logger = logging.getLogger("test.compile")
        rc = run(args, logger)

        assert rc == 0
        assert out_path.exists()
        result = yaml.safe_load(out_path.read_text())
        assert result["section"]["resolved"] is True
        assert "$ref" not in out_path.read_text()

    def test_compile_to_json_file(self, tmp_path):
        from fluid_build.cli.compile import run

        _write_yaml(tmp_path / "contract.yaml", {"name": "test", "val": 42})
        out_path = tmp_path / "bundled.json"

        args = argparse.Namespace(
            contract=str(tmp_path / "contract.yaml"),
            out=str(out_path),
            env=None,
            format=None,  # should infer from .json extension
        )
        logger = logging.getLogger("test.compile")
        rc = run(args, logger)

        assert rc == 0
        result = json.loads(out_path.read_text())
        assert result["name"] == "test"

    def test_compile_missing_file_returns_2(self, tmp_path):
        from fluid_build.cli.compile import run

        args = argparse.Namespace(
            contract=str(tmp_path / "nonexistent.yaml"),
            out="-",
            env=None,
            format=None,
        )
        logger = logging.getLogger("test.compile")
        rc = run(args, logger)
        assert rc == 2

    def test_compile_broken_ref_returns_2(self, tmp_path):
        from fluid_build.cli.compile import run

        _write_yaml(
            tmp_path / "contract.yaml",
            {"section": {"$ref": "./missing.yaml"}},
        )

        args = argparse.Namespace(
            contract=str(tmp_path / "contract.yaml"),
            out="-",
            env=None,
            format=None,
        )
        logger = logging.getLogger("test.compile")
        rc = run(args, logger)
        assert rc == 2

    def test_compile_with_env_overlay(self, tmp_path):
        from fluid_build.cli.compile import run

        _write_yaml(
            tmp_path / "contract.yaml",
            {"name": "test", "region": "us-dev"},
        )
        overlays = tmp_path / "overlays"
        overlays.mkdir()
        _write_yaml(overlays / "prod.yaml", {"region": "eu-prod"})

        out_path = tmp_path / "out.yaml"
        args = argparse.Namespace(
            contract=str(tmp_path / "contract.yaml"),
            out=str(out_path),
            env="prod",
            format=None,
        )
        logger = logging.getLogger("test.compile")
        rc = run(args, logger)

        assert rc == 0
        result = yaml.safe_load(out_path.read_text())
        assert result["region"] == "eu-prod"
