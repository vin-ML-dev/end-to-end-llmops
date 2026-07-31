"""Tiny client to exercise the gateway (non-streaming + streaming).

Usage (gateway must be running):
    python -m src.serving.client_demo
    python -m src.serving.client_demo --stream
"""

import argparse
import json
import os

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--prompt", default="What is the capital of Japan?")
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    headers = {}
    key = os.getenv("DOMAINBOT_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    body = {"messages": [{"role": "user", "content": args.prompt}], "stream": args.stream}

    if not args.stream:
        r = httpx.post(f"{args.url}/v1/chat", json=body, headers=headers, timeout=60)
        r.raise_for_status()
        d = r.json()
        print(f"[{d['model']} @ {d['revision']}]")
        print(d["content"])
        return

    with httpx.stream("POST", f"{args.url}/v1/chat", json=body, headers=headers, timeout=60) as r:
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                print()
                return
            try:
                print(json.loads(payload).get("content", ""), end="", flush=True)
            except json.JSONDecodeError:
                pass


if __name__ == "__main__":
    main()
