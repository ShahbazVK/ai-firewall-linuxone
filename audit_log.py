"""Encrypted audit trail for the AI Firewall.

Each request decision is serialized as one JSON object, encrypted with
AES-256-GCM, and appended to a local JSONL file. Every line carries its own
12-byte nonce and 16-byte authentication tag, so tampering or truncation is
detected when analytics reads the log back through :func:`read_events`.

The 256-bit encryption key is read from ``AUDIT_ENCRYPTION_KEY`` when set
(base64url or hex). Otherwise a key is generated once at ``.audit_key`` next to
this module and reused across restarts. The key file is created with owner-only
permissions.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Crypto.Cipher import AES

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", BASE_DIR / "audit.log.enc"))
KEY_PATH = Path(os.getenv("AUDIT_KEY_PATH", BASE_DIR / ".audit_key"))

_NONCE_BYTES = 12
_TAG_BYTES = 16
_AAD = b"ai-firewall-audit-v1"
_LOCK = threading.Lock()


def _decode_key(material: str) -> bytes:
    """Decode a 32-byte key supplied as base64url or hex."""
    material = material.strip()
    try:
        if len(material) == 64:
            key = bytes.fromhex(material)
        else:
            key = base64.urlsafe_b64decode(material.encode("utf-8"))
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Audit key is not valid base64url or hex") from exc

    if len(key) != 32:
        raise ValueError("Audit key must decode to exactly 32 bytes")
    return key


def _load_key() -> bytes:
    """Return the configured AES key, generating a local key if needed."""
    configured = os.getenv("AUDIT_ENCRYPTION_KEY")
    if configured:
        return _decode_key(configured)

    if KEY_PATH.exists():
        return _decode_key(KEY_PATH.read_text(encoding="utf-8"))

    key = os.urandom(32)
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(
        base64.urlsafe_b64encode(key).decode("ascii"),
        encoding="utf-8",
    )
    try:
        KEY_PATH.chmod(0o600)
    except OSError:
        logger.warning("Could not restrict permissions on %s", KEY_PATH)
    logger.info("Generated audit encryption key at %s", KEY_PATH)
    return key


def _key() -> bytes:
    """Load the key and fail clearly when the configuration is invalid."""
    try:
        return _load_key()
    except ValueError as exc:
        raise RuntimeError(
            "AUDIT_ENCRYPTION_KEY must be a 32-byte key encoded as base64url "
            "or 64 hex characters. Generate one with "
            "`python -c \"import base64,os; "
            "print(base64.urlsafe_b64encode(os.urandom(32)).decode())\"`."
        ) from exc


_KEY = _key()


def log_event(event: dict[str, Any]) -> dict[str, Any]:
    """Encrypt and append one audit event, returning the stored payload.

    ``event`` must be JSON-serializable. ``id`` and a UTC ``timestamp`` are
    added automatically. Writes are serialized so concurrent requests cannot
    interleave ciphertext lines.
    """
    payload: dict[str, Any] = {
        **event,
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    nonce = os.urandom(_NONCE_BYTES)
    cipher = AES.new(_KEY, AES.MODE_GCM, nonce=nonce)
    cipher.update(_AAD)
    ciphertext, tag = cipher.encrypt_and_digest(serialized)
    token = base64.urlsafe_b64encode(nonce + tag + ciphertext)

    with _LOCK:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        created = not LOG_PATH.exists()
        with LOG_PATH.open("ab") as handle:
            handle.write(token + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if created:
            try:
                LOG_PATH.chmod(0o600)
            except OSError:
                logger.warning("Could not restrict permissions on %s", LOG_PATH)

    return payload


def read_events() -> list[dict[str, Any]]:
    """Decrypt and return every readable audit event in append order.

    AES-GCM authenticates each record. A partially written, tampered, or
    corrupted line is skipped rather than breaking the analytics endpoint. The
    warning deliberately does not include the token or plaintext.
    """
    if not LOG_PATH.exists():
        return []

    with _LOCK:
        raw_lines = LOG_PATH.read_bytes().splitlines()

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        try:
            record = base64.urlsafe_b64decode(line)
            if len(record) < _NONCE_BYTES + _TAG_BYTES:
                raise ValueError("Audit record is too short")
            nonce = record[:_NONCE_BYTES]
            tag = record[_NONCE_BYTES : _NONCE_BYTES + _TAG_BYTES]
            ciphertext = record[_NONCE_BYTES + _TAG_BYTES :]
            cipher = AES.new(_KEY, AES.MODE_GCM, nonce=nonce)
            cipher.update(_AAD)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            event = json.loads(plaintext)
        except (ValueError, binascii.Error, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping unreadable audit event on line %s: %s",
                line_number,
                type(exc).__name__,
            )
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


# Convenient aliases for callers that prefer log-oriented names.
log_audit_event = log_event
read_logs = read_events
