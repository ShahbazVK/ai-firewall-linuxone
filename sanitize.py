import re

SECRET_PATTERNS = {
    "OPENAI_API_KEY": r"\bsk-proj-[A-Za-z0-9_-]{10,}\b",
    "AWS_ACCESS_KEY": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "GITHUB_TOKEN": r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
    "GITLAB_TOKEN": r"\bglpat-[A-Za-z0-9_-]{20,}\b",
}

def sanitize_code(code):
    sanitized = code
    detection_report = {}

    for secret_type, pattern in SECRET_PATTERNS.items():
        sanitized, count = re.subn(
            pattern,
            f"[REDACTED_{secret_type}]",
            sanitized
        )

        if count > 0:
            detection_report[secret_type] = count

    return {
        "sanitized_code": sanitized,
        "detections": detection_report,
        "total_secrets": sum(detection_report.values())
    }
