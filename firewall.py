"""PII redaction engine for the AI Firewall.

Scans text for Social Security numbers, email addresses, US phone numbers, and
16-digit credit card numbers, swaps each match for a labelled placeholder, and
returns the sanitised text together with a human-readable audit log.
"""

import re
from typing import TypedDict


class RedactionResult(TypedDict):
    """Shape returned by :func:`redact_pii`."""

    cleaned_text: str
    logs: list[str]


# XXX-XX-XXXX with word boundaries so longer digit runs are left alone.
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Local part @ domain with a dotted TLD of at least two letters.
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Exactly 16 digits, optionally grouped by spaces or dashes in blocks of four.
CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")

# US phone formats: (XXX) XXX-XXXX, XXX-XXX-XXXX, and XXX.XXX.XXXX.
# Parenthesized area codes may be followed by an optional space; the final
# separator accepts either a dash or a dot.
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\(\d{3}\)\s*|\d{3}[.-])\d{3}[.-]\d{4}(?!\d)"
)

# (singular label, plural label, pattern, replacement) — applied in order.
# Email runs first so a numeric local part is treated as an address rather than
# a card number.
_REDACTORS: tuple[tuple[str, str, re.Pattern[str], str], ...] = (
    ("email", "emails", EMAIL_PATTERN, "[REDACTED_EMAIL]"),
    ("phone", "phones", PHONE_PATTERN, "[REDACTED_PHONE]"),
    ("SSN", "SSNs", SSN_PATTERN, "[REDACTED_SSN]"),
    ("credit card", "credit cards", CREDIT_CARD_PATTERN, "[REDACTED_CREDIT_CARD]"),
)


def redact_pii(text: str) -> RedactionResult:
    """Replace PII in ``text`` with placeholders and describe what was removed.

    Returns a dictionary with ``cleaned_text`` (the redacted string) and
    ``logs`` (one message per PII type that matched, empty if none did).
    """
    cleaned_text = text
    logs: list[str] = []

    for singular, plural, pattern, replacement in _REDACTORS:
        cleaned_text, count = pattern.subn(replacement, cleaned_text)
        if count:
            label = singular if count == 1 else plural
            logs.append(f"Redacted {count} {label}")

    return {"cleaned_text": cleaned_text, "logs": logs}
