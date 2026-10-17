"""AI Firewall — FastAPI service.

This is the hackathon starting point: a thin API that accepts a chat prompt and
runs it through the firewall before forwarding it to an LLM.

The firewall currently redacts PII and credentials and blocks prompt-injection
attempts; the LLM itself is still mocked and echoes the cleaned text back to the
caller. The service also serves the chat UI at ``/`` and the analytics dashboard
at ``/admin``.
"""

from __future__ import annotations

import hmac
import io
import logging
import os
import re
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Header, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from audit_log import log_event, read_events
from firewall import redact_pii
from ml_guard import check_injection, explain_injection
from sanitize import SECRET_PATTERNS, sanitize_code

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "hackathon-secret")
RATE_LIMIT = "15/minute"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
LOCKDOWN_MESSAGE = "SYSTEM LOCKDOWN: All traffic blocked due to active threat."

# Process-local emergency stop. Run a single worker (or back this with shared
# state) if the deployment scales horizontally.
LOCKDOWN_MODE = False
_lockdown_lock = threading.Lock()

app = FastAPI(
    title="AI Firewall",
    description="Screens prompts before they reach an LLM.",
    version="0.1.0",
)

limiter = Limiter(key_func=get_remote_address, headers_enabled=True)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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


class PdfPolicyError(ValueError):
    """Rejection raised when a PDF fails an upload policy check."""


def _lockdown_response() -> JSONResponse:
    """Return the standard response used while lockdown mode is active."""
    return JSONResponse(
        status_code=503,
        content={"error": LOCKDOWN_MESSAGE, "lockdown": True},
    )


def _authorized_admin(admin_key: str | None) -> bool:
    """Check the admin header without leaking the configured key via timing."""
    if not admin_key:
        return False
    return hmac.compare_digest(
        admin_key.encode("utf-8"), ADMIN_API_KEY.encode("utf-8")
    )


@app.middleware("http")
async def enforce_lockdown(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Reject screened traffic before rate-limit accounting while locked down."""
    path = request.url.path.rstrip("/") or "/"
    if LOCKDOWN_MODE and path in {"/chat", "/upload"}:
        return _lockdown_response()
    return await call_next(request)


def _empty_pii_counts() -> dict[str, int]:
    """Return a zeroed PII counter for every supported entity type."""
    return {key: 0 for key in _PII_KEYS}


def _as_count(value: object) -> int:
    """Coerce a stored counter to a non-negative integer."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _empty_secret_counts() -> dict[str, int]:
    """Return a zeroed credential counter for every supported secret type."""
    return {key: 0 for key in SECRET_PATTERNS}


def _secret_counts(value: object) -> dict[str, int]:
    """Normalize stored credential-redaction counts into known integer keys."""
    counts = _empty_secret_counts()
    if not isinstance(value, dict):
        return counts
    for key in counts:
        counts[key] = _as_count(value.get(key))
    return counts


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
    source: str,
    suspicious_words: list[str] | None = None,
    confidence: float | None = None,
    secret_counts: dict[str, int] | None = None,
) -> None:
    """Persist one encrypted, redacted request event for analytics."""
    pii_counts = _pii_counts_from_logs(firewall_log)
    normalized_secrets = _secret_counts(secret_counts)
    event: dict[str, object] = {
        "kind": f"{source}_request",
        "source": source,
        "decision": decision,
        "pii_counts": pii_counts,
        "pii_total": sum(pii_counts.values()),
        "secret_counts": normalized_secrets,
        "secret_total": sum(normalized_secrets.values()),
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
    secret_totals = _empty_secret_counts()
    secret_requests = 0
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

        normalized_secrets = _secret_counts(event.get("secret_counts"))
        for key, count in normalized_secrets.items():
            secret_totals[key] += count
        if sum(normalized_secrets.values()) > 0:
            secret_requests += 1

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
        normalized_secrets = _secret_counts(event.get("secret_counts"))

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
                "secret_counts": normalized_secrets,
                "secret_total": sum(normalized_secrets.values()),
                "suspicious_words": words,
                "confidence": confidence if isinstance(confidence, (int, float)) else None,
            }
        )

    return {
        "total_requests": total,
        "pii_redacted": sum(pii_totals.values()),
        "pii_requests": pii_requests,
        "secrets_redacted": sum(secret_totals.values()),
        "secrets_requests": secret_requests,
        "injection_blocked": blocked,
        "passed": passed,
        "blocked": blocked,
        "block_rate": round((blocked / total) * 100, 1) if total else 0.0,
        "pii_types": pii_totals,
        "secret_types": secret_totals,
        "lockdown": LOCKDOWN_MODE,
        "recent_events": recent_events,
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }


def _screen_text(text: str, *, source: str) -> dict[str, object]:
    """Run untrusted text through PII, credential, and ML screening.

    ``source`` is stored in the encrypted audit event so chat and PDF traffic
    can be distinguished without storing either original payload.
    """
    result = redact_pii(text)
    pii_cleaned_text = result["cleaned_text"]

    sanitized_result = sanitize_code(pii_cleaned_text)
    cleaned_text = sanitized_result["sanitized_code"]
    secret_counts = _secret_counts(sanitized_result["detections"])
    firewall_log = list(result["logs"])

    for secret_type, count in secret_counts.items():
        if count:
            firewall_log.append(f"Redacted {count} {secret_type}")

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
            source=source,
            suspicious_words=suspicious_words,
            confidence=confidence,
            secret_counts=secret_counts,
        )
        return {
            "blocked": True,
            "error": "Blocked: Prompt Injection Detected",
            "cleaned_text": cleaned_text,
            "firewall_log": firewall_log,
            "confidence": confidence,
            "suspicious_words": suspicious_words,
        }

    firewall_log.append("ML check passed")
    _record_request(
        cleaned_text=cleaned_text,
        firewall_log=firewall_log,
        decision="passed",
        source=source,
        secret_counts=secret_counts,
    )
    return {
        "blocked": False,
        "cleaned_text": cleaned_text,
        "firewall_log": firewall_log,
        "llm_response": cleaned_text,
    }


def _extract_pdf_text(raw_pdf: bytes) -> tuple[str, int]:
    """Extract plain text and page count from an in-memory PDF."""
    reader = PdfReader(io.BytesIO(raw_pdf))
    if reader.is_encrypted:
        raise PdfPolicyError("Encrypted PDFs are not supported")

    pages = len(reader.pages)
    if pages == 0:
        raise PdfPolicyError("The PDF contains no pages")
    if pages > MAX_PDF_PAGES:
        raise PdfPolicyError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit")

    text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    return text, pages


@app.post("/api/lockdown", response_model=None)
def toggle_lockdown(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> dict[str, object] | JSONResponse:
    """Toggle the emergency traffic block. Requires the admin header."""
    global LOCKDOWN_MODE

    if not _authorized_admin(x_admin_key):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized: valid X-Admin-Key required"},
        )

    with _lockdown_lock:
        LOCKDOWN_MODE = not LOCKDOWN_MODE
        active = LOCKDOWN_MODE

    logger.warning("Lockdown mode %s", "enabled" if active else "disabled")
    return {
        "lockdown": active,
        "message": (
            LOCKDOWN_MESSAGE
            if active
            else "Lockdown cleared. Traffic is flowing normally."
        ),
    }


@app.post("/chat", response_model=None)
@limiter.shared_limit(RATE_LIMIT, scope="firewall")
def chat(
    request: Request, response: Response, payload: ChatRequest
) -> dict[str, object] | JSONResponse:
    """Run a prompt through the firewall and return the (mock) LLM reply.

    PII and credentials are redacted first, then the sanitized text is classified
    by the ML guard. Injections are rejected with a 400, top trigger words, and
    an audit log; they never reach the LLM. Safe prompts are echoed back by the
    mock LLM.
    """
    if LOCKDOWN_MODE:
        return _lockdown_response()

    screened = _screen_text(payload.text, source="chat")
    if screened["blocked"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": screened["error"],
                "cleaned_text": screened["cleaned_text"],
                "firewall_log": screened["firewall_log"],
                "confidence": screened["confidence"],
                "suspicious_words": screened["suspicious_words"],
            },
        )

    return {
        "original_text": payload.text,
        "cleaned_text": screened["cleaned_text"],
        "firewall_log": screened["firewall_log"],
        "llm_response": screened["llm_response"],
    }


@app.post("/upload", response_model=None)
@limiter.shared_limit(RATE_LIMIT, scope="firewall")
def upload_pdf(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
) -> dict[str, object] | JSONResponse:
    """Extract text from a PDF and screen it with the same firewall pipeline."""
    if LOCKDOWN_MODE:
        return _lockdown_response()

    filename = Path(file.filename or "").name or "upload.pdf"
    if (
        not filename.lower().endswith(".pdf")
        or file.content_type not in {"application/pdf", "application/x-pdf"}
    ):
        return JSONResponse(
            status_code=415,
            content={"error": "Only PDF files are supported"},
        )

    raw_pdf = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw_pdf:
        return JSONResponse(status_code=400, content={"error": "The PDF is empty"})
    if len(raw_pdf) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
            },
        )
    if not raw_pdf[:1024].lstrip().startswith(b"%PDF-"):
        return JSONResponse(
            status_code=415,
            content={"error": "The uploaded file is not a valid PDF"},
        )

    try:
        extracted_text, pages = _extract_pdf_text(raw_pdf)
    except PdfPolicyError as exc:
        # Known policy rejections (encrypted, page cap) are safe to explain.
        logger.warning("Rejected PDF upload %s: %s", filename, exc)
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        # PDF parsers raise a wide variety of exceptions on hostile input. Treat
        # any parse failure as a client error rather than leaking a 500 or
        # exposing parser internals.
        logger.warning(
            "Rejected PDF upload %s: %s",
            filename,
            type(exc).__name__,
            exc_info=True,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "Could not read PDF. The file may be corrupted or unsupported."
            },
        )

    if not extracted_text:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "No extractable text found. Scanned or image-only PDFs "
                    "are not supported."
                )
            },
        )

    screened = _screen_text(extracted_text, source="upload")
    body: dict[str, object] = {
        "filename": filename,
        "pages": pages,
        "cleaned_text": screened["cleaned_text"],
        "firewall_log": screened["firewall_log"],
    }
    if screened["blocked"]:
        body.update(
            {
                "error": screened["error"],
                "confidence": screened["confidence"],
                "suspicious_words": screened["suspicious_words"],
            }
        )
        return JSONResponse(status_code=400, content=body)

    body["llm_response"] = screened["llm_response"]
    return body

