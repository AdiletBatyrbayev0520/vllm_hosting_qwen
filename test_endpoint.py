"""
Pytest tests for the vLLM Qwen3-VL endpoint (OpenAI-compatible API on port 8000).

Run:
    pytest test_endpoint.py -v
    pytest test_endpoint.py -v -k "text"       # text-only tests
    pytest test_endpoint.py -v -k "image"      # vision tests
    pytest test_endpoint.py -v --timeout=120   # with custom timeout
"""

import base64
import io
import json
import os
from pathlib import Path

import pytest
import requests

BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "90"))

CHAT_URL = f"{BASE_URL}/v1/chat/completions"
MODELS_URL = f"{BASE_URL}/v1/models"

# Path to the real test image sitting next to this file
TEST_IMAGE_PATH = Path(__file__).parent / "test.jpg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chat(messages: list, **kwargs) -> requests.Response:
    payload = {"model": MODEL, "messages": messages, "max_tokens": 256, **kwargs}
    return requests.post(CHAT_URL, json=payload, timeout=TIMEOUT)


def assert_ok(resp: requests.Response):
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"
    data = resp.json()
    assert "choices" in data, f"No 'choices' in response: {data}"
    assert len(data["choices"]) > 0
    return data


def load_image_b64(path: Path) -> str:
    """Read an image file from disk and return its base64-encoded content."""
    return base64.b64encode(path.read_bytes()).decode()


def make_jpeg_bytes(width: int = 224, height: int = 224, color: tuple = (255, 0, 0)) -> bytes:
    """Generate a solid-color JPEG image in memory using Pillow."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def jpeg_base64(color: tuple = (255, 0, 0)) -> str:
    return base64.b64encode(make_jpeg_bytes(color=color)).decode()


# ---------------------------------------------------------------------------
# Health / model discovery
# ---------------------------------------------------------------------------

class TestHealth:
    def test_models_endpoint(self):
        resp = requests.get(MODELS_URL, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        ids = [m["id"] for m in data["data"]]
        assert any(MODEL in mid or mid in MODEL for mid in ids), (
            f"Expected model '{MODEL}' not found in {ids}"
        )

    def test_server_reachable(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        assert resp.status_code in (200, 404), (
            f"Server not reachable: HTTP {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# Text-only tests
# ---------------------------------------------------------------------------

class TestTextMessages:
    def test_simple_text(self):
        resp = chat([{"role": "user", "content": "Reply with exactly: OK"}])
        data = assert_ok(resp)
        text = data["choices"][0]["message"]["content"]
        assert isinstance(text, str) and len(text) > 0

    def test_system_and_user(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        data = assert_ok(chat(messages))
        assert "4" in data["choices"][0]["message"]["content"]

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "user", "content": "My name is Alex."},
            {"role": "assistant", "content": "Hello Alex!"},
            {"role": "user", "content": "What is my name?"},
        ]
        data = assert_ok(chat(messages))
        assert "Alex" in data["choices"][0]["message"]["content"]

    def test_temperature_zero(self):
        messages = [{"role": "user", "content": "What is 1 + 1? Answer with just the number."}]
        r1 = assert_ok(chat(messages, temperature=0))
        r2 = assert_ok(chat(messages, temperature=0))
        assert r1["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"]

    def test_max_tokens_respected(self):
        messages = [{"role": "user", "content": "Write a very long essay about the universe."}]
        data = assert_ok(chat(messages, max_tokens=10))
        tokens_used = data["usage"]["completion_tokens"]
        assert tokens_used <= 15, f"Expected ≤15 tokens, got {tokens_used}"

    def test_stop_sequence(self):
        messages = [{"role": "user", "content": "Count: one two three four five"}]
        data = assert_ok(chat(messages, stop=["three"], max_tokens=50))
        content = data["choices"][0]["message"]["content"]
        assert "four" not in content.lower()

    def test_response_structure(self):
        data = assert_ok(chat([{"role": "user", "content": "Hi"}]))
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "usage" in data
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert key in data["usage"]

    def test_n_equals_2(self):
        data = assert_ok(chat([{"role": "user", "content": "Say hi"}], n=2))
        assert len(data["choices"]) == 2

    def test_empty_system_message(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Say hello"},
        ]
        assert_ok(chat(messages))

    def test_long_prompt(self):
        long_text = "Summarize this: " + ("lorem ipsum dolor sit amet " * 100)
        messages = [{"role": "user", "content": long_text}]
        assert_ok(chat(messages, max_tokens=64))


# ---------------------------------------------------------------------------
# Vision / image tests
# ---------------------------------------------------------------------------

class TestImageMessages:
    def _image_message(self, b64: str, prompt: str = "What color is this image?") -> list:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def test_red_image(self):
        b64 = jpeg_base64(color=(255, 0, 0))
        data = assert_ok(chat(self._image_message(b64, "Is this image mostly red? Answer yes or no.")))
        content = data["choices"][0]["message"]["content"].lower()
        assert "yes" in content or "red" in content

    def test_blue_image(self):
        b64 = jpeg_base64(color=(0, 0, 255))
        data = assert_ok(chat(self._image_message(b64, "What color is this image? One word.")))
        assert isinstance(data["choices"][0]["message"]["content"], str)

    def test_green_image(self):
        b64 = jpeg_base64(color=(0, 255, 0))
        assert_ok(chat(self._image_message(b64, "Describe the image in one sentence.")))

    def test_multiple_images(self):
        red_b64 = jpeg_base64(color=(255, 0, 0))
        blue_b64 = jpeg_base64(color=(0, 0, 255))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{red_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{blue_b64}"}},
                    {"type": "text", "text": "How many images are there?"},
                ],
            }
        ]
        data = assert_ok(chat(messages))
        assert "2" in data["choices"][0]["message"]["content"] or "two" in data["choices"][0]["message"]["content"].lower()

    def test_image_with_system_prompt(self):
        b64 = jpeg_base64(color=(128, 128, 128))
        messages = [
            {"role": "system", "content": "You are a color expert. Always mention exact RGB values when possible."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]
        assert_ok(chat(messages))

    def test_image_then_text_followup(self):
        b64 = jpeg_base64(color=(255, 255, 0))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "What color is this?"},
                ],
            },
            {"role": "assistant", "content": "The image is yellow."},
            {"role": "user", "content": "Is yellow a warm or cool color?"},
        ]
        data = assert_ok(chat(messages))
        content = data["choices"][0]["message"]["content"].lower()
        assert "warm" in content or "cool" in content

    def test_malformed_base64_image(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,NOT_VALID!!!"}},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
        resp = chat(messages)
        assert resp.status_code in (200, 400, 422, 500), f"Unexpected status: {resp.status_code}"


# ---------------------------------------------------------------------------
# Real image tests (uses test.jpg from the project directory)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEST_IMAGE_PATH.exists(), reason="test.jpg not found")
class TestRealImage:
    """Tests using the actual test.jpg — an illustrated cover for 'Abay Zholy'."""

    @pytest.fixture(scope="class")
    def b64(self):
        return load_image_b64(TEST_IMAGE_PATH)

    def _msg(self, b64: str, prompt: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def test_describe_image(self, b64):
        data = assert_ok(chat(self._msg(b64, "Describe what you see in this image.")))
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 20

    def test_count_people(self, b64):
        data = assert_ok(chat(self._msg(b64, "How many people are in this image? Answer with a number.")))
        content = data["choices"][0]["message"]["content"]
        assert "3" in content or "three" in content.lower()

    def test_read_text_in_image(self, b64):
        data = assert_ok(chat(self._msg(b64, "What text is written at the bottom of this image?")))
        content = data["choices"][0]["message"]["content"]
        # The image contains Kazakh text — check the model returns something non-empty
        assert len(content) > 5

    def test_art_style(self, b64):
        data = assert_ok(chat(self._msg(b64, "Is this a photograph or an illustration? One word.")))
        content = data["choices"][0]["message"]["content"].lower()
        assert "illustration" in content or "drawing" in content or "art" in content or "painted" in content

    def test_image_with_followup(self, b64):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Describe the clothing of the central figure."},
                ],
            },
            {"role": "assistant", "content": "The central figure wears a teal/turquoise headband and a colorful robe."},
            {"role": "user", "content": "What does the headband color symbolize in Central Asian culture?"},
        ]
        assert_ok(chat(messages))

    def test_stream_real_image(self, b64):
        payload = {
            "model": MODEL,
            "messages": self._msg(b64, "What is happening in this image?"),
            "max_tokens": 128,
            "stream": True,
        }
        resp = requests.post(CHAT_URL, json=payload, stream=True, timeout=TIMEOUT)
        assert resp.status_code == 200
        chunks = []
        for line in resp.iter_lines():
            if line and line != b"data: [DONE]":
                raw = line.decode().removeprefix("data: ")
                chunk = json.loads(raw)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    chunks.append(delta)
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

class TestStreaming:
    def test_stream_text(self):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 64,
            "stream": True,
        }
        resp = requests.post(CHAT_URL, json=payload, stream=True, timeout=TIMEOUT)
        assert resp.status_code == 200

        chunks = []
        for line in resp.iter_lines():
            if line and line != b"data: [DONE]":
                raw = line.decode().removeprefix("data: ")
                chunk = json.loads(raw)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    chunks.append(delta)

        assert len(chunks) > 0, "No content chunks received in stream"
        full = "".join(chunks)
        assert any(d in full for d in ["1", "2", "3"])

    def test_stream_image(self):
        b64 = jpeg_base64(color=(255, 0, 0))
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": "What color is this?"},
                    ],
                }
            ],
            "max_tokens": 64,
            "stream": True,
        }
        resp = requests.post(CHAT_URL, json=payload, stream=True, timeout=TIMEOUT)
        assert resp.status_code == 200
        content = b""
        for line in resp.iter_lines():
            content += line
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_model(self):
        # vLLM defaults to the loaded model when model field is omitted
        payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        resp = requests.post(CHAT_URL, json=payload, timeout=10)
        assert resp.status_code in (200, 400, 422)

    def test_wrong_model_name(self):
        payload = {
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }
        resp = requests.post(CHAT_URL, json=payload, timeout=10)
        assert resp.status_code in (400, 404)

    def test_empty_messages(self):
        resp = chat([])
        assert resp.status_code in (400, 422)

    def test_invalid_temperature(self):
        resp = chat([{"role": "user", "content": "hi"}], temperature=999)
        assert resp.status_code in (200, 400, 422)  # server may clamp or reject

    def test_negative_max_tokens(self):
        resp = chat([{"role": "user", "content": "hi"}], max_tokens=-1)
        assert resp.status_code in (400, 422)

    def test_malformed_image_base64(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,NOT_VALID_BASE64!!!"}},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
        resp = chat(messages)
        assert resp.status_code in (200, 400, 422, 500)
