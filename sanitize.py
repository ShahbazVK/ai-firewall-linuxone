"""Deterministic credential redaction for the AI Firewall.

This module adds a second sanitization pass after PII redaction. It replaces
recognizable API keys and repository tokens with typed placeholders and reports
per-type counts so the caller can audit what was removed without storing the
secret itself.
"""

from __future__ import annotations

import re
from typing import TypedDict

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "OPENAI_API_KEY": re.compile(r"\bsk-proj-[A-Za-z0-9_-]{10,}\b"),
    "AWS_ACCESS_KEY": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GITHUB_TOKEN": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "GITLAB_TOKEN": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
}


class SanitizationResult(TypedDict):
    """Result of a credential-sanitization pass."""

    sanitized_code: str
    detections: dict[str, int]
    total_secrets: int


def sanitize_code(code: str) -> SanitizationResult:
    """Replace recognized credentials with typed placeholders.

    Args:
        code: Text that has already passed through PII redaction.

    Returns:
        The sanitized text, per-type match counts, and the total number of
        credentials removed.
    """
    sanitized = code
    detection_report: dict[str, int] = {}

    for secret_type, pattern in SECRET_PATTERNS.items():
        sanitized, count = pattern.subn(
            f"[REDACTED_{secret_type}]",
            sanitized,
        )
        if count > 0:
            detection_report[secret_type] = count

    return {
        "sanitized_code": sanitized,
        "detections": detection_report,
        "total_secrets": sum(detection_report.values()),
    }
