# Changelog

All notable changes to FLUID Forge CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Multi-file contract composition via `$ref`** — split contracts across multiple files using `$ref` pointers (OpenAPI-style). Refs resolve transparently in all commands (validate, plan, apply, execute). Supports external file refs, JSON pointer fragments (`file.yaml#/section`), nested/transitive refs, and subdirectory resolution. Circular reference detection and depth limiting protect against malformed contracts.
- **`fluid compile` command** — bundle a multi-file contract into a single resolved document. Supports `--out` for file output, `--env` for overlay application, and `--format json|yaml` for output format control. Equivalent to `swagger-cli bundle` for OpenAPI.
- **`compile_contract()` API** — programmatic entry point for $ref resolution in `fluid_build.loader`.
- **`RefResolutionError` exception** — clear error type for $ref failures (missing files, circular refs, invalid pointers).
- Multi-file contract example: `examples/0.7.1/bitcoin-multifile/` demonstrating team-owned fragments.
- "What is FLUID?" section in README
- CI status badge in README
- `.secrets.baseline` for detect-secrets scanning
- `.pre-commit-config.yaml` with detect-secrets, black, and ruff hooks
- `.github/CODEOWNERS` for code review ownership
- `THIRD_PARTY_LICENSES.md` documenting all dependency licenses
- `CHANGELOG.md` (this file)
- CLI smoke tests (`tests/cli/test_cli_smoke.py` — 28 tests)
- Config manager tests (`tests/test_config_manager.py` — 45 tests)
- `docs/HOW_IT_WORKS.md` — architecture overview of SDK + CLI

### Fixed
- Removed hardcoded default password (`'admin'`) in Apache Atlas connector — now requires explicit config or `ATLAS_USERNAME`/`ATLAS_PASSWORD` env vars
- Replaced 17 occurrences of hardcoded AWS account ID `123456789012` with `YOUR_AWS_ACCOUNT_ID` placeholder
- AWS Glue/SageMaker fallback role ARNs now use `self.account_id` instead of hardcoded values

### Removed
- `fluid_build/cli/admin.py` — internal admin/diagnostics command (not for public use)
- `fluid_build/cli/analytics.py` — internal telemetry module (dead code)

### Changed
- Bandit security scan in CI now blocks on failures (removed `|| true` fallback)

## [0.7.1] — 2026-01-30

### Added
- Multi-provider Airflow DAG generation (`fluid generate-airflow`)
- GCP code generators: Airflow, Dagster, Prefect
- Snowflake code generators: Airflow, Dagster, Prefect
- AWS code generators: Airflow, Dagster, Prefect
- Circular dependency detection in contract validation
- Provider SDK extraction (`fluid-provider-sdk` v0.1.0)
- Plugin-based provider discovery via entry points
- Local provider with DuckDB backend
- Policy engine (check, compile, apply)
- Blueprint system for data product templates
- ODPS/ODCS standard export support
- Marketplace and catalog connectors
- Comprehensive GitHub Actions CI (Python 3.9–3.12, ruff, black, bandit, coverage)
- Apache 2.0 license headers on all source files
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- Issue templates (bug report, feature request, provider request)
- PR template

### Performance
- Contract generation: 0.29–2.54ms per contract
- 828 tests passing at release

## [0.5.7] — 2025-08-15

### Added
- Initial public release
- Core CLI: `validate`, `plan`, `apply`
- GCP BigQuery provider
- Contract schema v0.5.7
- Basic Airflow DAG export

[Unreleased]: https://github.com/agentics-rising/fluid-forge-cli/compare/v0.7.1...HEAD
[0.7.1]: https://github.com/agentics-rising/fluid-forge-cli/compare/v0.5.7...v0.7.1
[0.5.7]: https://github.com/agentics-rising/fluid-forge-cli/releases/tag/v0.5.7
