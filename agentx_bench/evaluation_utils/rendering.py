"""Subject-aware case rendering."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .models import BenchmarkCase, Subject


def render_cases(cases: list[BenchmarkCase], subject: Subject) -> list[BenchmarkCase]:
    rendered = [render_case(case, subject) for case in cases]
    unresolved = [case.case_id for case in rendered if _contains_placeholder(case.to_payload())]
    if unresolved:
        raise ValueError(f"unresolved placeholders remain in rendered cases: {', '.join(unresolved[:10])}")
    return rendered


def render_case(case: BenchmarkCase, subject: Subject) -> BenchmarkCase:
    return replace(
        case,
        input_json=_render_value(case.input_json, subject),
        expected_json=_render_value(case.expected_json, subject),
        attributes_json=_render_value(case.attributes_json, subject),
    )


def _render_value(value: Any, subject: Subject) -> Any:
    if isinstance(value, str):
        replacements = {
            "${subject.teamName}": subject.team_name,
            "${subject.subjectId}": subject.subject_id,
            "${subject.subjectName}": subject.subject_name,
            "${subject.subjectType}": subject.subject_type,
            "${subject.agentMode}": subject.agent_mode,
        }
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        return value
    if isinstance(value, dict):
        return {key: _render_value(item, subject) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_value(item, subject) for item in value]
    return value


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "${" in value
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False
