"""AI Firewall — FastAPI service.

This is the hackathon starting point: a thin API that accepts a chat prompt and
runs it through the firewall before forwarding it to an LLM.

The firewall currently redacts PII and blocks prompt-injection attempts; the
LLM itself is still mocked and echoes the cleaned text back to the caller. The
service also serves the chat UI at ``/`` and the analytics dashboard at
``/admin``.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from audit_log import log_event, read_events
from firewall import redact_pii
from ml_guard import check_injection, explain_injection

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AI Firewall",
    description="Screens prompts before they reach an LLM.",
    version="0.1.0",
)

_PII_KEYS = ("ssn", "email", "credit_card")
_PII_LABEL_TO_KEY = {
    "SSN": "ssn",
    "SSNs": "ssn",
    "email": "email",
    "emails": "email",
    "credit card": "credit_card",
    "credit cards": "credit_card",
}
_REDACTION_PATTERN = re.compile(r"^Redacted (\d+) (.+)$")


class ChatRequest(BaseModel):
    """Payload accepted by the /chat endpoint."""

    text: str


def _empty_pii_counts() -> dict[str, int]:
    """Return a zeroed PII counter for every supported entity type."""
    return {key: 0 for key in _PII_KEYS}


def _as_count(value: object) -> int:
    """Coerce a stored counter to a non-negative integer."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _suspicious_words(value: object) -> list[str]:
    """Normalize stored classifier attributions into a bounded, unique list."""
    if not isinstance(value, list):
        return []

    words: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        word = " ".join(item.split())
        key = word.lower()
        if not word or key in seen:
            continue
        words.append(word)
        seen.add(key)
        if len(words) == 5:
            break
    return words


def _pii_counts_from_logs(logs: list[str]) -> dict[str, int]:
    """Convert firewall redaction messages into per-type entity counts."""
    counts = _empty_pii_counts()
    for entry in logs:
        match = _REDACTION_PATTERN.match(entry)
        if not match:
            continue
        key = _PII_LABEL_TO_KEY.get(match.group(2))
        if key:
            counts[key] += int(match.group(1))
    return counts


def _preview(text: str, limit: int = 180) -> str:
    """Collapse whitespace and truncate a prompt for the live event table."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _record_request(
    *,
    cleaned_text: str,
    firewall_log: list[str],
    decision: str,
    suspicious_words: list[str] | None = None,
    confidence: float | None = None,
) -> None:
    """Persist one encrypted, PII-safe request event for analytics."""
    pii_counts = _pii_counts_from_logs(firewall_log)
    event: dict[str, object] = {
        "kind": "chat_request",
        "decision": decision,
        "pii_counts": pii_counts,
        "pii_total": sum(pii_counts.values()),
        "preview": _preview(cleaned_text),
        "suspicious_words": (suspicious_words or [])[:5],
    }
    if confidence is not None:
        event["confidence"] = round(float(confidence), 6)
    log_event(event)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the security dashboard so the UI and API share one origin."""
    return FileResponse(BASE_DIR / "index.html")


@app.get("/admin", include_in_schema=False)
def admin_dashboard() -> FileResponse:
    """Serve the Security Operations Center analytics dashboard."""
    return FileResponse(BASE_DIR / "admin.html")


@app.get("/admin.js", include_in_schema=False)
def admin_script() -> FileResponse:
    """Serve the analytics dashboard controller."""
    return FileResponse(BASE_DIR / "admin.js", media_type="application/javascript")


@app.get("/api/analytics")
def analytics() -> dict[str, object]:
    """Aggregate decrypted audit events for the SOC dashboard."""
    events = read_events()

    pii_totals = _empty_pii_counts()
    pii_requests = 0
    blocked = 0

    for event in events:
        counts = event.get("pii_counts")
        if isinstance(counts, dict):
            normalized = {key: _as_count(counts.get(key)) for key in _PII_KEYS}
        else:
            log_entries = event.get("firewall_log")
            normalized = _pii_counts_from_logs(
                [str(entry) for entry in log_entries]
                if isinstance(log_entries, list)
                else []
            )

        for key, count in normalized.items():
            pii_totals[key] += count
        if sum(normalized.values()) > 0:
            pii_requests += 1
        if str(event.get("decision", "")).lower() == "blocked":
            blocked += 1

    total = len(events)
    passed = total - blocked

    newest_first = sorted(
        events,
        key=lambda event: str(event.get("timestamp", "")),
        reverse=True,
    )
    recent_events: list[dict[str, object]] = []
    for event in newest_first[:50]:
        counts = event.get("pii_counts")
        if isinstance(counts, dict):
            normalized = {key: _as_count(counts.get(key)) for key in _PII_KEYS}
        else:
            normalized = _empty_pii_counts()

        words = _suspicious_words(event.get("suspicious_words"))
        confidence = event.get("confidence")
        decision = str(event.get("decision", "passed")).lower()
        preview = str(event.get("preview") or event.get("cleaned_text") or "")

        # Older blocked records may predate stored attributions. Re-derive them
        # from the redacted preview so Explainability stays complete.
        if decision == "blocked" and not words and preview:
            try:
                words = _suspicious_words(explain_injection(preview))
            except Exception:
                words = []
        elif decision != "blocked":
            words = []

        recent_events.append(
            {
                "id": str(event.get("id", "")),
                "timestamp": str(event.get("timestamp", "")),
                "decision": decision,
                "preview": preview,
                "pii_counts": normalized,
                "pii_total": sum(normalized.values()),
                "suspicious_words": words,
                "confidence": confidence if isinstance(confidence, (int, float)) else None,
            }
        )

    return {
        "total_requests": total,
        "pii_redacted": sum(pii_totals.values()),
        "pii_requests": pii_requests,
        "injection_blocked": blocked,
        "passed": passed,
        "blocked": blocked,
        "block_rate": round((blocked / total) * 100, 1) if total else 0.0,
        "pii_types": pii_totals,
        "recent_events": recent_events,
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }


@app.post("/chat", response_model=None)
def chat(request: ChatRequest) -> dict[str, object] | JSONResponse:
    """Run a prompt through the firewall and return the (mock) LLM reply.

    PII is redacted first, then the cleaned text is classified by the ML guard.
    Injections are rejected with a 400, top trigger words, and an audit log; they
    never reach the LLM. Safe prompts are echoed back by the mock LLM.
    """
    result = redact_pii(request.text)
    cleaned_text = result["cleaned_text"]
    firewall_log = list(result["logs"])

    guard = check_injection(cleaned_text)
    if guard["is_injection"]:
        confidence = guard["confidence"]
        suspicious_words = explain_injection(cleaned_text)
        trigger_words = ", ".join(suspicious_words) if suspicious_words else "none"
        firewall_log.append(
            "Blocked: prompt injection detected "
            f"(trigger words: {trigger_words})"
        )
        _record_request(
            cleaned_text=cleaned_text,
            firewall_log=firewall_log,
            decision="blocked",
            suspicious_words=suspicious_words,
            confidence=confidence,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "Blocked: Prompt Injection Detected",
                "cleaned_text": cleaned_text,
                "firewall_log": firewall_log,
                "confidence": confidence,
                "suspicious_words": suspicious_words,
            },
        )

    firewall_log.append("ML check passed")
    llm_response = cleaned_text
    _record_request(
        cleaned_text=cleaned_text,
        firewall_log=firewall_log,
        decision="passed",
    )

    return {
        "original_text": request.text,
        "cleaned_text": cleaned_text,
        "firewall_log": firewall_log,
        "llm_response": llm_response,
    }


