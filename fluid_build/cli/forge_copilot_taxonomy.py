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

"""Shared copilot taxonomy and normalization helpers."""

from __future__ import annotations

__all__ = [
    "USE_CASE_CHOICES",
    "USE_CASE_LABELS",
    "clean_text",
    "canonicalize_use_case_text",
    "normalize_use_case",
    "format_use_case_label",
    "normalize_copilot_context",
]

import re
from typing import Any, Dict, List, Optional

USE_CASE_CHOICES: List[Dict[str, str]] = [
    {"label": "Analytics & BI", "value": "analytics"},
    {"label": "ETL / Data Pipelines", "value": "etl_pipeline"},
    {"label": "Streaming / Real-time", "value": "streaming"},
    {"label": "ML / Feature Engineering", "value": "ml_pipeline"},
    {"label": "Data Platform / Lakehouse", "value": "data_platform"},
    {"label": "Other / Not sure", "value": "other"},
]

USE_CASE_LABELS = {choice["value"]: choice["label"] for choice in USE_CASE_CHOICES}

USE_CASE_ALIASES = {
    "analytics": "analytics",
    "analytics and bi": "analytics",
    "analytics bi": "analytics",
    "business intelligence": "analytics",
    "bi": "analytics",
    "reporting": "analytics",
    "dashboard": "analytics",
    "dashboards": "analytics",
    "etl": "etl_pipeline",
    "etl pipeline": "etl_pipeline",
    "etl data pipelines": "etl_pipeline",
    "data pipeline": "etl_pipeline",
    "data pipelines": "etl_pipeline",
    "pipeline": "etl_pipeline",
    "pipelines": "etl_pipeline",
    "streaming": "streaming",
    "streaming real time": "streaming",
    "real time": "streaming",
    "realtime": "streaming",
    "real time analytics": "streaming",
    "ml": "ml_pipeline",
    "ml pipeline": "ml_pipeline",
    "ml feature engineering": "ml_pipeline",
    "machine learning": "ml_pipeline",
    "machine learning pipeline": "ml_pipeline",
    "machine learning model": "ml_pipeline",
    "feature engineering": "ml_pipeline",
    "data lake": "data_platform",
    "data lakes": "data_platform",
    "data lakehouse": "data_platform",
    "data platform": "data_platform",
    "data platform lakehouse": "data_platform",
    "lakehouse": "data_platform",
    "other": "other",
    "other not sure": "other",
    "not sure": "other",
    "not certain": "other",
    "unsure": "other",
}


def clean_text(value: Any) -> str:
    """Return a trimmed string for optional context values."""
    return str(value or "").strip()


def canonicalize_use_case_text(value: Any) -> str:
    """Return a comparison-friendly use-case string."""
    text = clean_text(value).lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_use_case(value: Any) -> Optional[str]:
    """Normalize use-case variants to stable internal values."""
    text = canonicalize_use_case_text(value)
    if not text:
        return None
    if text in USE_CASE_ALIASES:
        return USE_CASE_ALIASES[text]
    if "machine learning" in text or "feature engineering" in text or re.search(r"\bml\b", text):
        return "ml_pipeline"
    if "streaming" in text or "real time" in text or "realtime" in text:
        return "streaming"
    if "data platform" in text or "data lake" in text or "lakehouse" in text:
        return "data_platform"
    if (
        "etl" in text
        or "data pipeline" in text
        or "pipeline" in text
        or "cdc" in text
        or "sync" in text
    ):
        return "etl_pipeline"
    if (
        "analytics" in text
        or "reporting" in text
        or "dashboard" in text
        or "scorecard" in text
        or "business intelligence" in text
        or re.search(r"\bbi\b", text)
    ):
        return "analytics"
    if "other" in text or "not sure" in text or "unsure" in text:
        return "other"
    return None


def format_use_case_label(use_case: Any, use_case_other: Any = None) -> str:
    """Return the user-facing use-case label for display surfaces."""
    other_text = clean_text(use_case_other)
    canonical = normalize_use_case(use_case)
    if canonical == "other" and other_text:
        return other_text
    if canonical:
        return USE_CASE_LABELS.get(canonical, canonical.replace("_", " ").title())
    if other_text:
        return other_text
    raw = clean_text(use_case)
    return raw or USE_CASE_LABELS["analytics"]


def normalize_copilot_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize known copilot context fields without dropping unknown keys."""
    normalized = dict(context)
    use_case = normalize_use_case(normalized.get("use_case"))
    if use_case:
        normalized["use_case"] = use_case
    other_text = clean_text(normalized.get("use_case_other"))
    if other_text:
        normalized["use_case_other"] = other_text
    else:
        normalized.pop("use_case_other", None)
    return normalized
