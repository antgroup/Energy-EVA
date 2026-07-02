"""Dataclasses shared by AgentX Bench modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Subject:
    subject_id: str
    subject_name: str
    subject_type: str
    team_name: str
    agent_mode: str = "multi_agent"
    model: str | None = None
    version: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subjectId": self.subject_id,
            "subjectName": self.subject_name,
            "subjectType": self.subject_type,
            "teamName": self.team_name,
            "agentMode": self.agent_mode,
        }
        if self.model:
            payload["model"] = self.model
        if self.version:
            payload["version"] = self.version
        return payload


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    name: str
    input_json: dict[str, Any]
    expected_json: dict[str, Any]
    attributes_json: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any], line_no: int) -> "BenchmarkCase":
        for field_name in ("caseId", "inputJson", "expectedJson"):
            if field_name not in raw:
                raise ValueError(f"cases.jsonl line {line_no}: missing {field_name}")
        if not isinstance(raw["inputJson"], dict):
            raise ValueError(f"cases.jsonl line {line_no}: inputJson must be an object")
        if not isinstance(raw["expectedJson"], dict):
            raise ValueError(f"cases.jsonl line {line_no}: expectedJson must be an object")
        return cls(
            case_id=str(raw["caseId"]),
            name=str(raw.get("name") or raw["caseId"]),
            input_json=dict(raw["inputJson"]),
            expected_json=dict(raw["expectedJson"]),
            attributes_json=dict(raw.get("attributesJson") or {}),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "caseId": self.case_id,
            "caseName": self.name,
            "inputJson": self.input_json,
            "expectedJson": self.expected_json,
            "attributesJson": self.attributes_json,
        }


@dataclass(frozen=True)
class SuiteBundle:
    dataset_dir: Path
    suite: dict[str, Any]
    cases: list[BenchmarkCase]
    scoring: dict[str, Any]

    @property
    def suite_id(self) -> str:
        return str(self.suite.get("datasetId") or self.suite.get("suite") or self.dataset_dir.name)

    @property
    def suite_version(self) -> str:
        return str(self.suite.get("version") or "v1")

    @property
    def as_of_date(self) -> str:
        return str(self.suite.get("defaultAsOfDate") or "")

    @property
    def timezone(self) -> str:
        return str(self.suite.get("timezone") or "Asia/Shanghai")

    def suite_payload(self) -> dict[str, Any]:
        return {
            "suiteId": self.suite_id,
            "suiteVersion": self.suite_version,
            "asOfDate": self.as_of_date,
            "timezone": self.timezone,
        }


@dataclass(frozen=True)
class RunContext:
    run_id: str
    suite: SuiteBundle
    subject: Subject
    metric_codes: list[str]
    output_root: Path


TERMINAL_STATUSES = {
    "COMPLETED",
    "COMPLETED_WITH_SKIPPED_METRICS",
    "FAILED",
    "VALIDATION_FAILED",
    "UNKNOWN_OR_EXPIRED",
    "TIMEOUT",
    "INCOMPLETE",
}

SUCCESS_CASE_STATUSES = {"SUCCESS", "COMPLETED"}
