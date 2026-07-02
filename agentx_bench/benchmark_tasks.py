#!/usr/bin/env python3
"""Inspect AgentX Bench datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx_bench.evaluation_utils.case_loader import load_suite, resolve_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List AgentX Bench tasks.")
    parser.add_argument("--suite", default="power_trade_standard_v1")
    parser.add_argument("--dataset")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    suite = load_suite(resolve_dataset(args.suite, args.dataset))
    if args.json:
        print(json.dumps({
            "suiteId": suite.suite_id,
            "suiteVersion": suite.suite_version,
            "caseCount": len(suite.cases),
            "cases": [case.to_payload() for case in suite.cases],
        }, ensure_ascii=False, indent=2))
        return 0
    print(f"{suite.suite_id} ({len(suite.cases)} cases)")
    for case in suite.cases:
        print(f"{case.case_id}\t{case.name}\t{case.attributes_json.get('caseType', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
