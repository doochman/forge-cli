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

"""Schema inference for CSV, JSON, Parquet, and Avro files."""

from __future__ import annotations

__all__ = [
    "summarize_sample_file",
    "read_parquet_metadata",
    "read_avro_metadata",
    "extract_provider_hints",
    "infer_scalar_type",
    "infer_python_type",
    "merge_types",
    "map_inferred_type_to_contract_type",
]

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

MAX_SAMPLE_ROWS = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_sample_file(path: Path) -> Dict[str, Any]:
    """Extract schema-only metadata from a data file (CSV, JSON, Parquet, Avro)."""
    suffix = path.suffix.lower()
    columns: Dict[str, str] = {}
    sampled_rows = 0
    row_count: Optional[int] = None
    schema_source: Optional[str] = None
    warnings: List[str] = []

    if suffix == ".csv":
        columns, sampled_rows = _infer_csv_schema(path)
    elif suffix in {".json", ".jsonl"}:
        columns, sampled_rows = _infer_json_schema(path)
    elif suffix in {".parquet", ".pq"}:
        metadata = read_parquet_metadata(path)
        columns = metadata.get("columns") or {}
        row_count = metadata.get("row_count")
        schema_source = metadata.get("schema_source")
        warnings = list(metadata.get("warnings") or [])
    elif suffix == ".avro":
        metadata = read_avro_metadata(path)
        columns = metadata.get("columns") or {}
        row_count = metadata.get("row_count")
        schema_source = metadata.get("schema_source")
        warnings = list(metadata.get("warnings") or [])

    summary: Dict[str, Any] = {
        "path": str(path),
        "format": suffix.lstrip("."),
        "sampled_rows": sampled_rows,
        "columns": columns,
        "provider_hints": extract_provider_hints(path.name),
    }
    if row_count is not None:
        summary["row_count"] = row_count
    if schema_source:
        summary["schema_source"] = schema_source
    if warnings:
        summary["warnings"] = warnings
    return summary


def read_parquet_metadata(path: Path) -> Dict[str, Any]:
    """Extract column schema from a Parquet file using pyarrow or duckdb."""
    for reader in (_read_parquet_metadata_pyarrow, _read_parquet_metadata_duckdb):
        try:
            metadata = reader(path)
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001
            return {
                "columns": {},
                "warnings": [f"Could not inspect Parquet schema for {path.name}: {exc}"],
            }
        if metadata.get("columns"):
            return metadata

    return {
        "columns": {},
        "warnings": [
            f"Parquet file {path.name} was discovered but schema extraction requires pyarrow or duckdb."
        ],
    }


def read_avro_metadata(path: Path) -> Dict[str, Any]:
    """Extract column schema from an Avro file using fastavro or avro."""
    for reader in (_read_avro_metadata_fastavro, _read_avro_metadata_avro):
        try:
            metadata = reader(path)
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001
            return {
                "columns": {},
                "warnings": [f"Could not inspect Avro schema for {path.name}: {exc}"],
            }
        if metadata.get("columns"):
            return metadata

    return {
        "columns": {},
        "warnings": [
            f"Avro file {path.name} was discovered but schema extraction requires fastavro or avro."
        ],
    }


def extract_provider_hints(text: str) -> List[str]:
    """Extract cloud provider hints from text (filenames, content, etc.)."""
    lowered = text.lower()
    hints = []
    if any(token in lowered for token in ("gcp", "bigquery", "composer", "dataform")):
        hints.append("gcp")
    if any(token in lowered for token in ("aws", "s3", "redshift", "athena", "glue")):
        hints.append("aws")
    if "snowflake" in lowered:
        hints.append("snowflake")
    if not hints and any(token in lowered for token in ("csv", "json", "local", "duckdb")):
        hints.append("local")
    return hints


def merge_types(values: Sequence[str]) -> str:
    """Pick the most common non-null type from a sequence of inferred types."""
    filtered = [value for value in values if value != "null"]
    if not filtered:
        return "string"
    return Counter(filtered).most_common(1)[0][0]


def map_inferred_type_to_contract_type(value: str) -> str:
    """Map an inferred type name to a FLUID contract column type."""
    mapping = {
        "boolean": "boolean",
        "integer": "integer",
        "number": "number",
        "date": "date",
        "datetime": "timestamp",
        "array": "array",
        "object": "object",
        "string": "string",
    }
    return mapping.get(value, "string")


def infer_scalar_type(value: Any) -> str:
    """Infer a scalar type from a string value (CSV cells, etc.)."""
    if value is None:
        return "null"
    text = str(value).strip()
    if text == "":
        return "null"
    if text.lower() in {"true", "false"}:
        return "boolean"
    if re.fullmatch(r"-?\d+", text):
        return "integer"
    if re.fullmatch(r"-?\d+\.\d+", text):
        return "number"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return "date"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-Zz]+", text):
        return "datetime"
    return "string"


def infer_python_type(value: Any) -> str:
    """Infer a type from a native Python value (JSON rows, etc.)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return infer_scalar_type(value)


# ---------------------------------------------------------------------------
# CSV / JSON inference
# ---------------------------------------------------------------------------


def _infer_csv_schema(path: Path) -> tuple[Dict[str, str], int]:
    sampled_rows = 0
    type_tracker: Dict[str, List[str]] = {}
    with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sampled_rows += 1
            for key, value in row.items():
                if key is None:
                    continue
                type_tracker.setdefault(key, []).append(infer_scalar_type(value))
            if sampled_rows >= MAX_SAMPLE_ROWS:
                break
    columns = {key: merge_types(values) for key, values in type_tracker.items()}
    return columns, sampled_rows


def _infer_json_schema(path: Path) -> tuple[Dict[str, str], int]:
    rows = list(load_json_rows(path))
    type_tracker: Dict[str, List[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key, value in row.items():
            type_tracker.setdefault(key, []).append(infer_python_type(value))
    columns = {key: merge_types(values) for key, values in type_tracker.items()}
    return columns, len(rows)


def load_json_rows(path: Path) -> Iterable[Any]:
    """Load rows from a JSON or JSONL file."""
    content = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".jsonl":
        for line in content.splitlines():
            if not line.strip():
                continue
            yield json.loads(line)
        return

    parsed = json.loads(content)
    if isinstance(parsed, list):
        for item in parsed[:MAX_SAMPLE_ROWS]:
            yield item
        return
    if isinstance(parsed, dict):
        if all(isinstance(value, list) for value in parsed.values()):
            keys = list(parsed.keys())
            row_count = min(len(parsed[key]) for key in keys)
            for index in range(min(row_count, MAX_SAMPLE_ROWS)):
                yield {key: parsed[key][index] for key in keys}
            return
        yield parsed


# ---------------------------------------------------------------------------
# Parquet readers
# ---------------------------------------------------------------------------


def _read_parquet_metadata_pyarrow(path: Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    columns = {field.name: infer_arrow_type(str(field.type)) for field in schema}
    row_count = parquet_file.metadata.num_rows if parquet_file.metadata else None
    return {"columns": columns, "row_count": row_count, "schema_source": "pyarrow"}


def _read_parquet_metadata_duckdb(path: Path) -> Dict[str, Any]:
    import duckdb

    connection = duckdb.connect()
    try:
        rows = connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
    finally:
        connection.close()
    columns = {
        str(row[0]): infer_duckdb_type(str(row[1]))
        for row in rows
        if len(row) >= 2 and row[0] is not None
    }
    return {"columns": columns, "schema_source": "duckdb"}


# ---------------------------------------------------------------------------
# Avro readers
# ---------------------------------------------------------------------------


def _read_avro_metadata_fastavro(path: Path) -> Dict[str, Any]:
    from fastavro import reader

    with path.open("rb") as handle:
        avro_reader = reader(handle)
        schema = avro_reader.writer_schema or {}
    return {"columns": extract_avro_columns(schema), "schema_source": "fastavro", "row_count": None}


def _read_avro_metadata_avro(path: Path) -> Dict[str, Any]:
    from avro.datafile import DataFileReader
    from avro.io import DatumReader

    with path.open("rb") as handle:
        reader = DataFileReader(handle, DatumReader())
        try:
            schema = json.loads(str(reader.datum_reader.writers_schema))
        finally:
            reader.close()
    return {"columns": extract_avro_columns(schema), "schema_source": "avro", "row_count": None}


def extract_avro_columns(schema: Mapping[str, Any]) -> Dict[str, str]:
    """Extract column names and types from an Avro schema record."""
    fields = schema.get("fields") or []
    columns: Dict[str, str] = {}
    for field in fields:
        name = field.get("name")
        if not name:
            continue
        columns[str(name)] = infer_avro_type(field.get("type"))
    return columns


# ---------------------------------------------------------------------------
# Type inference helpers
# ---------------------------------------------------------------------------


def infer_avro_type(type_spec: Any) -> str:
    """Map an Avro type spec to a simplified FLUID type."""
    if isinstance(type_spec, list):
        non_null = [c for c in type_spec if c != "null"]
        if not non_null:
            return "string"
        return infer_avro_type(non_null[0])
    if isinstance(type_spec, str):
        lowered = type_spec.lower()
        if lowered in {"boolean"}:
            return "boolean"
        if lowered in {"int", "long"}:
            return "integer"
        if lowered in {"float", "double"}:
            return "number"
        if lowered in {"bytes", "string", "enum"}:
            return "string"
        if lowered in {"array"}:
            return "array"
        if lowered in {"map", "record"}:
            return "object"
        return "string"
    if isinstance(type_spec, Mapping):
        logical_type = str(type_spec.get("logicalType") or "").lower()
        if logical_type in {"date"}:
            return "date"
        if logical_type in {
            "timestamp-millis",
            "timestamp-micros",
            "local-timestamp-millis",
            "local-timestamp-micros",
        }:
            return "datetime"
        avro_type = type_spec.get("type")
        if avro_type == "array":
            return "array"
        if avro_type in {"map", "record"}:
            return "object"
        if avro_type == "enum":
            return "string"
        return infer_avro_type(avro_type)
    return "string"


def infer_arrow_type(type_name: str) -> str:
    """Map a PyArrow type name to a simplified FLUID type."""
    lowered = type_name.lower()
    if "bool" in lowered:
        return "boolean"
    if any(token in lowered for token in ("int", "uint")):
        return "integer"
    if any(token in lowered for token in ("float", "double", "decimal")):
        return "number"
    if "timestamp" in lowered:
        return "datetime"
    if "date" in lowered:
        return "date"
    if any(token in lowered for token in ("list", "large_list", "fixed_size_list")):
        return "array"
    if any(token in lowered for token in ("struct", "map")):
        return "object"
    return "string"


def infer_duckdb_type(type_name: str) -> str:
    """Map a DuckDB type name to a simplified FLUID type."""
    lowered = type_name.lower()
    if "bool" in lowered:
        return "boolean"
    if any(token in lowered for token in ("tinyint", "smallint", "integer", "bigint", "hugeint")):
        return "integer"
    if any(token in lowered for token in ("float", "double", "decimal", "real")):
        return "number"
    if "timestamp" in lowered:
        return "datetime"
    if lowered == "date":
        return "date"
    if lowered.endswith("[]") or "list" in lowered or lowered.startswith("array"):
        return "array"
    if "struct" in lowered or "map" in lowered:
        return "object"
    return "string"
