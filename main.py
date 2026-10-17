"""AI Firewall — FastAPI service.

This is the hackathon starting point: a thin API that accepts a chat prompt and
runs it through the firewall before forwarding it to an LLM.

The firewall currently redacts PII and blocks prompt-injection attempts; the
LLM itself is still mocked and echoes the cleaned text back to the caller.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from firewall import redact_pii
from ml_guard import check_injection
from sanitize import sanitize_code

app = FastAPI(
    title="AI Firewall",
    description="Screens prompts before they reach an LLM.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    """Payload accepted by the /chat endpoint."""

    text: str


@app.post("/chat", response_model=None)
def chat(request: ChatRequest) -> dict[str, object] | JSONResponse:
    """Run a prompt through the firewall and return the (mock) LLM reply.

    PII is redacted first, then the cleaned text is classified by the ML guard.
    Injections are rejected with a 400 and never reach the LLM; safe prompts are
    echoed back by the mock LLM.
    """
    result = redact_pii(request.text)
    cleaned_text = result["cleaned_text"]

    """My regex-based sanitizer for redacting secrets like API keys, tokens, etc."""

    sanitized_result = sanitize_code(cleaned_text)
    final_text = sanitized_result["sanitized_code"]
    firewall_log = list(result["logs"])

    guard = check_injection(final_text)  
    if guard["is_injection"]:
        confidence = guard["confidence"]
        firewall_log.append(
            f"Blocked: prompt injection detected (confidence {confidence:.2f})"
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "Blocked: Prompt Injection Detected",
                "cleaned_text": cleaned_text,
                "firewall_log": firewall_log,
                "confidence": confidence,
            },
        )

    firewall_log.append("ML check passed")
    llm_response = cleaned_text

    return {
        "original_text": request.text,
        "cleaned_text": cleaned_text,
        "firewall_log": firewall_log,
        "llm_response": llm_response,
    }


