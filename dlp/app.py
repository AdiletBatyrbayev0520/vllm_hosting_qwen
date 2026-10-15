"""DLP gateway: an OpenAI-compatible proxy that checks every prompt before it
reaches the LLM provider.

    client -> POST /v1/chat/completions -> [regex rules] -> [Laya] -> allow -> provider
                                                                   -> block -> refusal message

Regex rules catch structured secrets (keys, tokens, card numbers) that a
classifier can miss; Laya catches semantic leaks (internal financials, client
data, unreleased plans). Fails closed: if Laya is unreachable, the prompt is blocked.
"""

import json
import logging
import os
import re
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

PROVIDER_URL = os.getenv("PROVIDER_URL", "http://qwen3-vl-server:8000/v1").rstrip("/")
PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY", "")
LAYA_URL = os.getenv("LAYA_URL", "http://laya-server:8002").rstrip("/")
LAYA_API_KEY = os.getenv("LAYA_API_KEY", "")
BLOCK_THRESHOLD = float(os.getenv("DLP_BLOCK_THRESHOLD", "0.5"))

log = logging.getLogger("dlp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = FastAPI(title="DLP gateway")


# --- Rule layer -------------------------------------------------------------

RULES = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "api_token": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}"
        r"|hf_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})\b"
    ),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "password_assignment": re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key)\s*[:=]\s*['\"]?[^\s'\"]{6,}"
    ),
}
CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def rule_hits(text: str) -> list[str]:
    hits = [name for name, rx in RULES.items() if rx.search(text)]
    for m in CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            hits.append("payment_card")
            break
    return hits


# --- Classifier layer -------------------------------------------------------

CATEGORIES = {
    "credentials": "passwords, API keys, tokens, private keys, connection strings",
    "personal_data": "personal data of customers or employees: IDs, addresses, phone numbers, medical or salary records",
    "financial": "non-public financial results, budgets, bank details, deal terms",
    "source_code": "proprietary source code or internal system configuration",
    "business_secrets": "unreleased products, strategy, contracts, internal documents marked confidential",
    "none": "nothing confidential; general questions or public information",
}
QUESTIONS = {
    "leak": {
        "type": "noul",
        "instructions": "Does this message disclose confidential or sensitive company, "
        "customer, or employee information that must not be sent to an external AI service?",
    },
    "category": {
        "type": "choice",
        "instructions": "What kind of sensitive information does this message contain?",
        "criteria": CATEGORIES,
    },
}


async def classify(client: httpx.AsyncClient, text: str) -> dict:
    headers = {"Authorization": f"Bearer {LAYA_API_KEY}"} if LAYA_API_KEY else {}
    r = await client.post(
        f"{LAYA_URL}/v1/systemone",
        json={"state": {"message": text}, "questions": QUESTIONS},
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    answers = r.json()["answers"]
    return {"leak_probability": answers["leak"]["noul"], "category": answers["category"]["choice"]}


# --- Decision ---------------------------------------------------------------


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal: keep only the text parts
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


async def inspect(client: httpx.AsyncClient, messages: list[dict]) -> dict:
    user_texts = [_text_of(m) for m in messages if m.get("role") == "user"]
    # Rules are cheap, so they scan the whole conversation the client is about to send.
    hits = sorted({h for t in user_texts for h in rule_hits(t)})
    if hits:
        return {"action": "block", "reason": "rule", "detail": hits}
    # Laya's English checkpoint reads 512 tokens, so it judges the newest user turn;
    # earlier turns were judged when they were new.
    latest = next((t for t in reversed(user_texts) if t.strip()), "")
    if not latest:
        return {"action": "allow", "reason": "empty"}
    try:
        verdict = await classify(client, latest)
    except Exception:
        log.exception("Laya unavailable; failing closed")
        return {"action": "block", "reason": "classifier_unavailable"}
    blocked = verdict["leak_probability"] >= BLOCK_THRESHOLD and verdict["category"] != "none"
    return {"action": "block" if blocked else "allow", "reason": "classifier", "detail": verdict}


def refusal_text(decision: dict) -> str:
    if decision["reason"] == "classifier_unavailable":
        return ("⚠️ Your message was not sent: the data-loss-prevention check is unavailable "
                "right now. Please try again later.")
    if decision["reason"] == "rule":
        what = ", ".join(h.replace("_", " ") for h in decision["detail"])
    else:
        what = decision["detail"]["category"].replace("_", " ")
    return (f"🚫 Your message was blocked and was NOT sent to the AI provider: it appears to "
            f"contain confidential information ({what}). Please remove or anonymize the "
            f"sensitive data and try again.")


def _completion(model: str, text: str, stream: bool):
    cid, created = f"chatcmpl-dlp-{uuid.uuid4().hex[:12]}", int(time.time())
    if not stream:
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "finish_reason": "content_filter",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }, headers={"X-DLP-Decision": "block"})

    def events():
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'content_filter'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"X-DLP-Decision": "block"})


# --- Routes -----------------------------------------------------------------

client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))


def _provider_headers() -> dict:
    return {"Authorization": f"Bearer {PROVIDER_API_KEY}"} if PROVIDER_API_KEY else {}


@app.get("/health")
def health():
    return {"status": "ok", "provider": PROVIDER_URL, "classifier": LAYA_URL}


@app.get("/v1/models")
async def models():
    try:
        r = await client.get(f"{PROVIDER_URL}/models", headers=_provider_headers())
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"provider unreachable at {PROVIDER_URL}: {e!r}")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/v1/dlp/check")
async def check(body: dict):
    """Inspect without forwarding, e.g. {"messages": [{"role": "user", "content": "..."}]}."""
    return await inspect(client, body.get("messages", []))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be a list")

    decision = await inspect(client, messages)
    # Log the decision, never the prompt itself: the log must not become the leak.
    log.info("dlp %s", json.dumps(decision))
    if decision["action"] == "block":
        return _completion(body.get("model", "dlp-gateway"), refusal_text(decision),
                           bool(body.get("stream")))

    url, headers = f"{PROVIDER_URL}/chat/completions", _provider_headers()
    if not body.get("stream"):
        r = await client.post(url, json=body, headers=headers)
        return JSONResponse(r.json(), status_code=r.status_code,
                            headers={"X-DLP-Decision": "allow"})

    req = client.build_request("POST", url, json=body, headers=headers)
    upstream = await client.send(req, stream=True)

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code,
                             media_type="text/event-stream", headers={"X-DLP-Decision": "allow"})
