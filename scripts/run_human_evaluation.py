"""Capture reproducible responses from the fact-check service for human review."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.base_url, timeout=60) as client, args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            case_id = str(row.get("case_id", "")).strip()
            message = str(row.get("input", "")).strip()
            if not case_id or not message:
                raise ValueError("cada caso precisa de case_id e input")
            response = client.post("/api/v2/debunk", json={"message": message})
            response.raise_for_status()
            output.write(
                json.dumps(
                    {
                        **row,
                        "captured_at": datetime.now(UTC).isoformat(),
                        "response": response.json(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
