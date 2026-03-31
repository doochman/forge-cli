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

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

from fluid_build.cli.console import cprint
from fluid_build.cli.console import error as console_error

from ..policy.agent_policy import validate_agent_policy
from ..policy.sovereignty import validate_sovereignty
from ..schema_manager import FluidSchemaManager, SchemaVersion, ValidationResult, VersionConstraint
from ..structured_logging import (
    log_metric,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ._common import CLIError, load_contract_with_overlay
from ._logging import error, info, warn

COMMAND = "validate"


def register(subparsers: argparse._SubParsersAction):
    p = subparsers.add_parser(
        COMMAND,
        help="Validate a FLUID contract against official schemas",
        description="""
        Enhanced FLUID contract validation with dynamic schema fetching,
        version detection, and comprehensive error reporting.
        
        This command automatically detects the FLUID version in your contract
        and validates against the appropriate schema from the official repository.
        Schemas are cached locally for offline use.
        """,
        epilog="""
Examples:
  # Basic validation
  fluid validate contract.fluid.yaml

  # Validate with environment overlay
  fluid validate contract.fluid.yaml --env prod

  # Verbose validation with schema info
  fluid validate contract.fluid.yaml --verbose --show-schema

  # Validate against specific schema version
  fluid validate contract.fluid.yaml --schema-version 0.5.7

  # Strict validation (warnings as errors)
  fluid validate contract.fluid.yaml --strict
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    p.add_argument("contract", nargs="?", help="Path to contract.fluid.(yaml|json)")

    # Optional arguments
    p.add_argument("--env", help="Overlay environment (dev/test/prod)")

    # Version control
    p.add_argument(
        "--schema-version",
        help="Specific schema version to validate against (e.g., '0.4.0', '0.5.0')",
    )
    p.add_argument("--min-version", help="Minimum acceptable schema version (e.g., '>=0.4.0')")
    p.add_argument("--max-version", help="Maximum acceptable schema version (e.g., '<0.6.0')")

    # Validation options
    p.add_argument("--strict", action="store_true", default=False, help="Treat warnings as errors")
    p.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Only use cached/bundled schemas (no network access)",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Force refresh of cached schemas",
    )

    # Cache management
    p.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Clear schema cache before validation",
    )
    p.add_argument("--cache-dir", type=Path, help="Custom cache directory for schemas")

    # Output options
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Verbose output with detailed validation info",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true", default=False, help="Minimal output (errors only)"
    )
    p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # Schema information
    p.add_argument(
        "--list-versions",
        action="store_true",
        default=False,
        help="List available schema versions and exit",
    )
    p.add_argument(
        "--show-schema",
        action="store_true",
        default=False,
        help="Show the schema being used for validation",
    )

    p.set_defaults(cmd=COMMAND, func=run)


def run(args, logger: logging.Logger) -> int:
    """Enhanced validation command with comprehensive schema management."""
    start_time = time.time()

    try:
        # Log operation start
        log_operation_start(
            logger,
            "validate_contract",
            contract=str(args.contract) if args.contract else None,
            env=getattr(args, "env", None),
            strict=args.strict,
        )

        # Initialize schema manager
        schema_manager = FluidSchemaManager(cache_dir=args.cache_dir, logger=logger)

        # Handle cache clearing
        if args.clear_cache:
            removed = schema_manager.clear_cache()
            if not args.quiet:
                cprint(f"Cleared {removed} cached schema files")
            log_metric(logger, "schemas_cleared", removed, unit="files")

        # Handle list versions
        if args.list_versions:
            return _handle_list_versions(schema_manager, args, logger)

        # Handle case where no contract is provided but required
        if not args.contract:
            raise CLIError(
                1,
                "contract_required",
                {"message": "Contract file is required unless using --list-versions"},
            )

        # Validate contract file existence
        contract_path = Path(args.contract)
        if not contract_path.exists():
            raise CLIError(1, "contract_file_not_found", {"path": str(contract_path)})

        # Load contract with overlay
        try:
            contract = load_contract_with_overlay(args.contract, getattr(args, "env", None), logger)
        except Exception as e:
            raise CLIError(1, "contract_load_failed", {"error": str(e)})

        # Determine target schema version
        target_version, auto_selected = _determine_target_version(contract, args, schema_manager, logger)

        # Validate version constraints
        _validate_version_constraints(target_version, args, logger)

        # Perform validation, with one-step fallback for auto-selected latest versions
        target_version, validation_result = _validate_with_version_fallback(
            contract=contract,
            target_version=target_version,
            auto_selected=auto_selected,
            args=args,
            schema_manager=schema_manager,
            logger=logger,
        )

        # Show schema if requested
        if args.show_schema:
            _show_schema_info(target_version, schema_manager, args, logger)

        # Log metrics
        duration = time.time() - start_time
        log_metric(logger, "validation_duration", duration, unit="seconds")
        log_metric(logger, "validation_errors", len(validation_result.errors), unit="count")
        log_metric(logger, "validation_warnings", len(validation_result.warnings), unit="count")

        # Output results
        exit_code = _output_results(validation_result, args, logger)

        # Log operation result
        if exit_code == 0:
            log_operation_success(
                logger,
                "validate_contract",
                duration=duration,
                schema_version=str(target_version),
                valid=validation_result.is_valid,
            )
        else:
            log_operation_failure(
                logger,
                "validate_contract",
                error=f"Validation failed with {len(validation_result.errors)} errors",
                duration=duration,
            )

        return exit_code

    except CLIError as e:
        duration = time.time() - start_time
        log_operation_failure(logger, "validate_contract", error=e.event, duration=duration)

        # Handle specific CLI errors with user-friendly messages
        if e.event == "version_below_minimum":
            if not args.quiet:
                console_error(
                    f"Contract version {e.context.get('version')} does not meet minimum requirement {e.context.get('constraint')}"
                )
        elif e.event == "version_above_maximum":
            if not args.quiet:
                console_error(
                    f"Contract version {e.context.get('version')} exceeds maximum allowed {e.context.get('constraint')}"
                )
        elif e.event == "contract_file_not_found":
            if not args.quiet:
                console_error(f"Contract file not found: {e.context.get('path')}")
        elif e.event == "contract_required":
            if not args.quiet:
                console_error(f"{e.context.get('message', 'Contract file is required')}")
        else:
            if not args.quiet:
                console_error(f"Validation error: {e.event}")
                if e.context:
                    for key, value in e.context.items():
                        cprint(f"   {key}: {value}")

        return e.exit_code
    except Exception as e:
        raise CLIError(1, "cli_unhandled_exception", {"error": str(e)})


def _handle_list_versions(schema_manager: FluidSchemaManager, args, logger: logging.Logger) -> int:
    """Handle --list-versions flag."""
    try:
        versions = schema_manager.list_available_versions(include_remote=not args.offline)

        if args.format == "json":
            import json

            cprint(json.dumps({"available_versions": versions}, indent=2))
        else:
            cprint("Available FLUID Schema Versions:")
            cprint("==================================")

            bundled = schema_manager.BUNDLED_VERSIONS
            cached = schema_manager.cache.list_cached_versions()

            for version in versions:
                status_indicators = []
                if version in bundled:
                    status_indicators.append("bundled")
                if version in cached:
                    status_indicators.append("cached")

                status = f" ({', '.join(status_indicators)})" if status_indicators else ""
                cprint(f"  {version}{status}")

            if not args.offline:
                cprint("\nNote: Additional versions may be available remotely.")
                cprint("Use --offline to see only local versions.")

        return 0

    except Exception as e:
        error(logger, "list_versions_failed", {"error": str(e)})
        return 1


def _determine_target_version(
    contract: dict, args, schema_manager: FluidSchemaManager, logger: logging.Logger
) -> Tuple[Optional[SchemaVersion], bool]:
    """Determine which schema version to validate against."""

    # Explicit version specified
    if args.schema_version:
        try:
            return SchemaVersion.parse(args.schema_version), False
        except ValueError as e:
            raise CLIError(
                1, "invalid_schema_version", {"version": args.schema_version, "error": str(e)}
            )

    # Auto-detect from contract
    detected = schema_manager.detect_version(contract)
    if detected:
        if args.verbose:
            info(logger, f"Detected FLUID version: {detected}")
        return detected, False

    default_version = _find_latest_compatible_version(args, schema_manager)
    warn(logger, f"No fluidVersion detected, defaulting to latest compatible version: {default_version}")
    return default_version, True


def _available_schema_versions(schema_manager: FluidSchemaManager, args) -> list[SchemaVersion]:
    versions = schema_manager.list_available_versions(include_remote=not args.offline)
    return [SchemaVersion.parse(version) for version in versions]


def _filter_compatible_versions(versions: list[SchemaVersion], args) -> list[SchemaVersion]:
    compatible = versions

    if args.min_version:
        try:
            min_constraint = VersionConstraint.parse(args.min_version)
        except ValueError as e:
            raise CLIError(1, "invalid_min_version", {"version": args.min_version, "error": str(e)})
        compatible = [version for version in compatible if min_constraint.matches(version)]

    if args.max_version:
        try:
            max_constraint = VersionConstraint.parse(args.max_version)
        except ValueError as e:
            raise CLIError(1, "invalid_max_version", {"version": args.max_version, "error": str(e)})
        compatible = [version for version in compatible if max_constraint.matches(version)]

    return compatible


def _find_latest_compatible_version(args, schema_manager: FluidSchemaManager) -> SchemaVersion:
    versions = _available_schema_versions(schema_manager, args)
    compatible_versions = _filter_compatible_versions(versions, args)
    if compatible_versions:
        return compatible_versions[-1]

    if versions:
        return versions[-1]

    return SchemaVersion.parse("0.5.7")


def _find_previous_compatible_version(
    current_version: SchemaVersion, args, schema_manager: FluidSchemaManager
) -> Optional[SchemaVersion]:
    versions = _available_schema_versions(schema_manager, args)
    compatible_versions = _filter_compatible_versions(versions, args)
    previous_versions = [version for version in compatible_versions if version < current_version]
    return previous_versions[-1] if previous_versions else None


def _validate_contract_for_version(
    contract: dict,
    target_version: Optional[SchemaVersion],
    args,
    schema_manager: FluidSchemaManager,
    logger: logging.Logger,
) -> ValidationResult:
    validation_result = schema_manager.validate_contract(
        contract, schema_version=target_version, strict=args.strict, offline_only=args.offline
    )

    # FLUID 0.7.1+ governance validation
    if target_version and target_version >= SchemaVersion.parse("0.7.1"):
        if not args.quiet and args.verbose:
            info(logger, "Running FLUID 0.7.1 governance validation...")

        sovereignty_valid, sovereignty_messages = validate_sovereignty(contract)
        for msg in sovereignty_messages:
            if "❌" in msg:
                validation_result.add_error(msg)
            elif "⚠️" in msg:
                validation_result.add_warning(msg)
            else:
                if args.verbose:
                    info(logger, msg)

        if not sovereignty_valid:
            validation_result.is_valid = False

        agent_policy_valid, agent_messages = validate_agent_policy(contract)
        for msg in agent_messages:
            if "❌" in msg:
                validation_result.add_error(msg)
            elif "⚠️" in msg:
                validation_result.add_warning(msg)
            else:
                if args.verbose:
                    info(logger, msg)

        if not agent_policy_valid:
            validation_result.is_valid = False

    return validation_result


def _validate_with_version_fallback(
    contract: dict,
    target_version: Optional[SchemaVersion],
    auto_selected: bool,
    args,
    schema_manager: FluidSchemaManager,
    logger: logging.Logger,
) -> Tuple[Optional[SchemaVersion], ValidationResult]:
    validation_result = _validate_contract_for_version(
        contract=contract,
        target_version=target_version,
        args=args,
        schema_manager=schema_manager,
        logger=logger,
    )

    if not auto_selected or validation_result.is_valid or not target_version:
        return target_version, validation_result

    previous_version = _find_previous_compatible_version(target_version, args, schema_manager)
    if not previous_version:
        return target_version, validation_result

    warn(
        logger,
        f"Validation failed for auto-selected version {target_version}; retrying previous compatible version {previous_version}",
    )
    fallback_result = _validate_contract_for_version(
        contract=contract,
        target_version=previous_version,
        args=args,
        schema_manager=schema_manager,
        logger=logger,
    )
    return previous_version, fallback_result


def _validate_version_constraints(
    version: Optional[SchemaVersion], args, logger: logging.Logger
) -> None:
    """Validate that the target version meets constraints."""
    if not version:
        return

    # Check minimum version constraint
    if args.min_version:
        try:
            min_constraint = VersionConstraint.parse(args.min_version)
            if not min_constraint.matches(version):
                raise CLIError(
                    2,
                    "version_below_minimum",
                    {"version": str(version), "constraint": args.min_version},
                )
        except ValueError as e:
            raise CLIError(1, "invalid_min_version", {"version": args.min_version, "error": str(e)})

    # Check maximum version constraint
    if args.max_version:
        try:
            max_constraint = VersionConstraint.parse(args.max_version)
            if not max_constraint.matches(version):
                raise CLIError(
                    2,
                    "version_above_maximum",
                    {"version": str(version), "constraint": args.max_version},
                )
        except ValueError as e:
            raise CLIError(1, "invalid_max_version", {"version": args.max_version, "error": str(e)})


def _show_schema_info(
    version: Optional[SchemaVersion],
    schema_manager: FluidSchemaManager,
    args,
    logger: logging.Logger,
) -> None:
    """Show information about the schema being used."""
    if not version:
        return

    schema = schema_manager.get_schema(version, offline_only=args.offline)
    if not schema:
        warn(logger, f"Schema not available for version {version}")
        return

    if args.format == "json":
        import json

        cprint("Schema Information:")
        cprint("==================")
        cprint(json.dumps(schema, indent=2))
    else:
        cprint(f"\nSchema Information for v{version}:")
        cprint("=" * 40)
        cprint(f"Version: {version}")
        cprint(f"Schema URL: {version.schema_url}")

        # Show schema metadata if available
        if "$schema" in schema:
            cprint(f"JSON Schema: {schema['$schema']}")
        if "title" in schema:
            cprint(f"Title: {schema['title']}")
        if "description" in schema:
            cprint(f"Description: {schema['description']}")

        cprint()


def _output_results(result: ValidationResult, args, logger: logging.Logger) -> int:
    """Output validation results in the requested format."""

    if args.format == "json":
        return _output_json_results(result, args)
    else:
        return _output_text_results(result, args, logger)


def _output_json_results(result: ValidationResult, args) -> int:
    """Output results in JSON format."""
    import json

    output = {
        "valid": result.is_valid,
        "schema_version": str(result.schema_version) if result.schema_version else None,
        "errors": result.errors,
        "warnings": result.warnings,
        "validation_time": result.validation_time,
    }

    cprint(json.dumps(output, indent=2))
    return 0 if result.is_valid and (not args.strict or not result.warnings) else 1


def _output_text_results(result: ValidationResult, args, logger: logging.Logger) -> int:
    """Output results in human-readable text format."""

    # Summary
    if not args.quiet:
        cprint(result.get_summary())
        cprint()

    # Errors
    if result.errors:
        if not args.quiet:
            cprint("Validation Errors:")
            cprint("==================")
        for i, error in enumerate(result.errors, 1):
            if args.quiet:
                cprint(f"ERROR: {error}")
            else:
                cprint(f"{i:2}. {error}")

        if not args.quiet:
            cprint()

    # Warnings
    if result.warnings and not args.quiet:
        cprint("Validation Warnings:")
        cprint("====================")
        for i, warning in enumerate(result.warnings, 1):
            cprint(f"{i:2}. {warning}")
        cprint()

    # Verbose information
    if args.verbose and not args.quiet:
        cprint("Validation Details:")
        cprint("===================")
        cprint(f"Schema Version: {result.schema_version}")
        cprint(f"Validation Time: {result.validation_time:.3f}s")
        cprint(f"Error Count: {len(result.errors)}")
        cprint(f"Warning Count: {len(result.warnings)}")

    # Determine exit code
    has_errors = bool(result.errors)
    has_warnings = bool(result.warnings)
    treat_warnings_as_errors = args.strict and has_warnings

    if has_errors or treat_warnings_as_errors:
        return 1
    else:
        return 0
