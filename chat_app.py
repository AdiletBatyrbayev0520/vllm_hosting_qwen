"""
Streamlit chat UI for the local Qwen3-VL vLLM endpoint.

Run:
    streamlit run chat_app.py
"""

import base64
import io
import json
import os

import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Qwen3-VL Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
    }

    /* Chat message bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
        backdrop-filter: blur(8px);
    }

    /* User bubble accent */
    [data-testid="stChatMessage"][data-testid*="user"] {
        border-left: 3px solid #7c3aed !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(124,58,237,0.4) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.25) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.45) !important;
    }

    /* Sliders & selects */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #7c3aed !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px;
    }

    /* Header title */
    .hero-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.4);
        margin-top: 2px;
        margin-bottom: 20px;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .status-online  { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
    .status-offline { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

    /* Image preview */
    .img-preview {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.12);
        max-height: 180px;
        object-fit: cover;
    }

    /* Scrollbar */
    ::-webkit-scrollbar       { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, image_b64?}]
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None   # base64 string of uploaded image


# ── Helpers ─────────────────────────────────────────────────────────────────────
def encode_image(uploaded_file) -> str:
    """Return a base64 data-URI string from a Streamlit UploadedFile."""
    raw = uploaded_file.read()
    b64 = base64.b64encode(raw).decode()
    mime = uploaded_file.type or "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_api_messages(history: list) -> list:
    """Convert session messages → OpenAI API message format."""
    api_msgs = []
    for msg in history:
        if msg["role"] == "system":
            api_msgs.append({"role": "system", "content": msg["content"]})
        elif msg.get("image_b64"):
            api_msgs.append({
                "role": msg["role"],
                "content": [
                    {"type": "image_url", "image_url": {"url": msg["image_b64"]}},
                    {"type": "text", "text": msg["content"]},
                ],
            })
        else:
            api_msgs.append({"role": msg["role"], "content": msg["content"]})
    return api_msgs


def check_server(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def stream_chat(base_url, model, messages, temperature, max_tokens, top_p):
    """Stream from the vLLM OpenAI-compatible endpoint; yield text chunks."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        usage = {}
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                # Capture usage if present (some vLLM versions send it in last chunk)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta, usage
        yield "", usage  # final yield to flush usage


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title">⚙️ Config</p>', unsafe_allow_html=True)

    base_url = st.text_input("Server URL", value="http://localhost:8000")
    model = st.text_input("Model", value="Qwen/Qwen3-VL-8B-Instruct")

    # Server status
    is_online = check_server(base_url)
    pill_cls  = "status-online" if is_online else "status-offline"
    pill_txt  = "● Online" if is_online else "● Offline"
    st.markdown(
        f'<span class="status-pill {pill_cls}">{pill_txt}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens  = st.slider("Max tokens",  64,  4096, 512, 64)
    top_p       = st.slider("Top-p",       0.1, 1.0,  0.9, 0.05)

    st.divider()

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful multimodal assistant. Be concise and accurate.",
        height=100,
    )

    st.divider()

    # Image uploader lives in sidebar so it persists across turns
    uploaded = st.file_uploader(
        "📎 Attach image to next message",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        key="image_uploader",
    )
    if uploaded:
        st.session_state.pending_image = encode_image(uploaded)
        st.image(uploaded, caption="Preview", use_container_width=True)
    else:
        st.session_state.pending_image = None

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Stats
    st.divider()
    st.metric("💬 Turns", len([m for m in st.session_state.messages if m["role"] == "user"]))
    st.metric("🔢 Tokens used", st.session_state.total_tokens)


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🤖 Qwen3-VL Chat</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Powered by vLLM · OpenAI-compatible API · localhost:8000</p>', unsafe_allow_html=True)

if not is_online:
    st.warning("⚠️ vLLM server is not reachable. Make sure `docker compose up` is running.")

# Render history
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        # Show attached image thumbnail if any
        if msg.get("image_b64"):
            try:
                header, b64data = msg["image_b64"].split(",", 1)
                raw = base64.b64decode(b64data)
                st.image(io.BytesIO(raw), width=240)
            except Exception:
                pass
        st.markdown(msg["content"])

# ── Chat input ──────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Type a message… (attach an image via the sidebar first if needed)"
)

if user_input:
    if not is_online:
        st.error("Cannot send — server is offline.")
        st.stop()

    image_b64 = st.session_state.pending_image

    # Inject system prompt at start of every fresh conversation
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "system", "content": system_prompt})

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "image_b64": image_b64,
    })

    # Show user message immediately
    with st.chat_message("user"):
        if image_b64:
            try:
                _, b64data = image_b64.split(",", 1)
                raw = base64.b64decode(b64data)
                st.image(io.BytesIO(raw), width=240)
            except Exception:
                pass
        st.markdown(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply  = ""
        last_usage  = {}

        try:
            api_msgs = build_api_messages(st.session_state.messages)
            for chunk, usage in stream_chat(
                base_url, model, api_msgs, temperature, max_tokens, top_p
            ):
                full_reply += chunk
                placeholder.markdown(full_reply + "▌")
                if usage:
                    last_usage = usage

            placeholder.markdown(full_reply)

        except requests.exceptions.ConnectionError:
            full_reply = "❌ Connection refused. Is the vLLM server running?"
            placeholder.error(full_reply)
        except requests.exceptions.Timeout:
            full_reply = "⏱ Request timed out."
            placeholder.error(full_reply)
        except Exception as e:
            full_reply = f"❌ Error: {e}"
            placeholder.error(full_reply)

    # Save assistant turn
    st.session_state.messages.append({"role": "assistant", "content": full_reply})

    # Update token counter
    if last_usage.get("total_tokens"):
        st.session_state.total_tokens += last_usage["total_tokens"]

    # Clear the pending image after sending
    st.session_state.pending_image = None
    st.rerun()
