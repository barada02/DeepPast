"""Quick connectivity test for OpenAI-compatible endpoint (DigitalOcean serverless).

Usage:
1) Put key in .env (project root):
   DO_AI_API_KEY=your_key
   DO_AI_BASE_URL=https://inference.do-ai.run/v1   # optional
   DO_AI_MODEL=llama3-8b-instruct                  # optional

2) Run from project root:
   python scripts/test_model_endpoint.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI


def load_env_file() -> None:
    """Load key=value pairs from .env if present."""
    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    load_env_file()

    api_key = os.getenv("DO_AI_API_KEY", "")
    base_url = os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1")
    model = os.getenv("DO_AI_MODEL", "openai-gpt-oss-20b")

    if not api_key:
        print("❌ Missing DO_AI_API_KEY. Add it to .env or environment variables.")
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with exactly: ENDPOINT_OK"},
            ],
            temperature=0,
            max_completion_tokens=256,
        )
    except Exception as exc:
        print(f"❌ Endpoint call failed: {exc}")
        return 2

    content = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)

    print("✅ Endpoint call succeeded")
    print(f"base_url: {base_url}")
    print(f"model: {model}")
    print(f"response: {content}")
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)
        print(f"usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

    if "ENDPOINT_OK" in content:
        print("✅ Content check passed")
        return 0

    print("⚠️ Endpoint worked, but content check did not match exact phrase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
