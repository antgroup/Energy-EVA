"""Terminal progress, diagnostics, and dashboard rendering for AgentX Bench."""

from __future__ import annotations

import platform
import shutil
import sys
import textwrap
import time
import os
from pathlib import Path
from typing import Any, TextIO

from agentx_bench import __version__

from .models import BenchmarkCase, RunContext


TERMINAL_DISPLAY_STATUSES = {
    "COMPLETED",
    "COMPLETED_WITH_SKIPPED_METRICS",
    "FAILED",
    "VALIDATION_FAILED",
    "UNKNOWN_OR_EXPIRED",
    "TIMEOUT",
    "INCOMPLETE",
    "INTERRUPTED",
    "CANCELED",
}

STAGES = [
    ("LOCAL_PREPARING", "Prepare"),
    ("SUBMITTED", "Accepted"),
    ("RUNNING", "Running"),
    ("SUBMITTING_CASES", "Submit cases"),
    ("COLLECTING_TRACE", "Collect trace"),
    ("COMPUTING_METRICS", "Metrics"),
    ("FETCHING_RESULTS", "Download"),
    ("RESULTS_READY", "Score"),
]

STAGE_INDEX = {status: index for index, (status, _) in enumerate(STAGES)}

STAGE_PROGRESS = {
    "LOCAL_PREPARING": 0.02,
    "SUBMITTED": 0.12,
    "RUNNING": 0.24,
    "SUBMITTING_CASES": 0.42,
    "COLLECTING_TRACE": 0.64,
    "COMPUTING_METRICS": 0.82,
    "FETCHING_RESULTS": 1.0,
    "RESULTS_READY": 1.0,
}

STAGE_GUIDANCE = {
    "LOCAL_PREPARING": "AgentX Bench is preparing the suite, selected cases, subject identity, and metric codes.",
    "SUBMITTED": "The Evaluation Service accepted the request and created an evalRunId.",
    "RUNNING": "The backend worker started the shared evaluation workflow.",
    "SUBMITTING_CASES": "Selected cases are being submitted into the Energent Swarm main flow.",
    "COLLECTING_TRACE": "Cases may already be answered; the service is collecting trace evidence for scoring.",
    "COMPUTING_METRICS": "Quality judges and exact calculators are turning answers and traces into metricResults.",
    "FETCHING_RESULTS": "AgentX Bench is downloading case results, metricResults, and evidence snapshots.",
    "RESULTS_READY": "Results are local; AgentX Bench is computing scores and writing report artifacts.",
}

ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "muted": "\x1b[2m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
}

AGENTX_BANNER = [
    " █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗  ██╗    ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗",
    "██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝╚██╗██╔╝    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║",
    "███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║    ╚███╔╝     ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║",
    "██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║    ██╔██╗     ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║",
    "██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██╔╝ ██╗    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║",
    "╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝",
]
AGENTX_BANNER_WIDTH = max(len(line) for line in AGENTX_BANNER)


class TerminalProgressReporter:
    def __init__(self, enabled: bool = True, stream: TextIO | None = None, mode: str = "auto",
                 color: str = "auto", input_stream: TextIO | None = None):
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.input_stream = input_stream or sys.stdin
        self.mode = mode
        self.color = color
        self.started_at = time.monotonic()
        self.selected_cases = 0
        self.last_status_signature = ""
        self.last_status_changed_at = self.started_at
        self.last_warning_bucket = -1
        self.panel_lines = 0
        self.context: RunContext | None = None
        self.method = ""
        self.service_url = ""
        self.timeout_seconds: float | None = None
        self.poll_interval_seconds: float | None = None
        self.run_dir: Path | None = None
        self.interactive = False
        self.color_enabled = False
        self.prompt_enabled = False

    def start(self, context: RunContext, selected_cases: int, method: str, run_dir: Path,
              service_url: str | None = None, timeout_seconds: float | None = None,
              poll_interval_seconds: float | None = None,
              cases: list[BenchmarkCase] | None = None) -> None:
        self.started_at = time.monotonic()
        self.last_status_changed_at = self.started_at
        self.selected_cases = selected_cases
        self.context = context
        self.method = method
        self.service_url = service_url or ""
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.run_dir = run_dir
        self.interactive = self._resolve_interactive()
        self.color_enabled = self._resolve_color()
        self.prompt_enabled = self._resolve_prompt_enabled()
        self._write_start_banner(context, selected_cases, method, run_dir, self.service_url)
        if cases:
            self._write_case_plan(cases)
        self._write_start_hint()

    def confirm_start(self) -> bool:
        if not self.enabled or not self.prompt_enabled:
            return True
        self.stream.write("Press Enter to start evaluation, or q to cancel: ")
        self.stream.flush()
        try:
            answer = self.input_stream.readline()
        except KeyboardInterrupt:
            self.stream.write("\n")
            self.stream.flush()
            return False
        if answer is None:
            return True
        return answer.strip().lower() != "q"

    def status(self, payload: dict[str, Any]) -> None:
        status_value = str(payload.get("status") or "UNKNOWN")
        signature = progress_signature(payload)
        changed = signature != self.last_status_signature
        now = time.monotonic()
        if changed:
            self.last_status_signature = signature
            self.last_status_changed_at = now
            self.last_warning_bucket = -1
        stable_for = now - self.last_status_changed_at
        diagnostics = build_run_diagnostics(
            payload,
            selected_cases=self.selected_cases,
            stable_for_seconds=stable_for,
        )
        if self.interactive:
            self._render_panel(payload, diagnostics, stable_for)
            return
        if changed or self._should_emit_waiting_warning(stable_for):
            self._write(self._status_line(payload, status_value, stable_for))
            for line in self._case_progress_lines(payload, compact=True):
                self._write(line)
            message = str(payload.get("message") or payload.get("errorMessage") or "").strip()
            if message:
                self._write(f"  message : {self._muted(message)}")
            for warning in diagnostics["warnings"][:3]:
                self._write(f"  warning : {self._warn(warning)}")

    def results_ready(self, normalized_results: dict[str, Any]) -> None:
        case_results = normalized_results.get("caseResults") or []
        total = len(case_results) or self.selected_cases
        done = sum(1 for row in case_results if str(row.get("status") or "").upper() in {"SUCCESS", "COMPLETED"})
        failed = max(0, total - done)
        status_value = str(normalized_results.get("status") or "UNKNOWN")
        if self.interactive:
            self._clear_panel()
        self._write(
            f"[results] {status_value} | cases {done}/{total} succeeded, {failed} failed/incomplete "
            f"| elapsed {format_duration(time.monotonic() - self.started_at)}"
        )

    def complete(self, summary: dict[str, Any], run_dir: Path, dashboard_enabled: bool = True) -> bool:
        if self.interactive:
            self._clear_panel()
        if not self.enabled or not dashboard_enabled:
            return True
        status_value = str(summary.get("status") or "UNKNOWN")
        selected = _int(summary.get("selectedCases"))
        completed = _int(summary.get("completedCases"))
        failed = _int(summary.get("failedCases"))
        incomplete = _int(summary.get("incompleteCases"))
        not_completed = max(selected - completed, failed + incomplete, 0)
        self._write("")
        self._write(self._title("AgentX Bench Calculation Complete"))
        self._write(
            f"Status    : {self._status(status_label(status_value), status_value)} ({status_value})"
        )
        self._write(
            f"Cases     : {completed}/{selected} completed, {failed} failed, {incomplete} incomplete, "
            f"{not_completed} not completed"
        )
        self._write(f"Elapsed   : {format_duration(time.monotonic() - self.started_at)}")
        self._write(f"Artifacts : {run_dir}")
        if self.prompt_enabled:
            self.stream.write("Press Enter to view final score dashboard...")
            self.stream.flush()
            try:
                self.input_stream.readline()
            except KeyboardInterrupt:
                self.stream.write("\n")
                self.stream.flush()
                return False
            self.stream.write("\n")
            self.stream.flush()
        return True

    def dashboard(self, text: str) -> None:
        if self.interactive:
            self._clear_panel()
        self.stream.write("\n")
        self.stream.write(text.rstrip())
        self.stream.write("\n")
        self.stream.flush()

    def _status_line(self, payload: dict[str, Any], status_value: str, stable_for: float) -> str:
        selected = _int(payload.get("selectedCaseCount")) or self.selected_cases
        submitted = _int(payload.get("submittedCaseCount"))
        case_runs = _int(payload.get("caseRunCount"))
        visible_case_runs = _known_case_count(payload)
        failed = _int(payload.get("failedCaseCount"))
        fraction = progress_fraction(payload, self.selected_cases)
        bar = progress_bar(fraction)
        phase = str(payload.get("phase") or "-")
        metric = _metric_label(payload)
        eval_run_id = str(payload.get("evalRunId") or "-")
        elapsed = format_duration(time.monotonic() - self.started_at)
        return (
            f"[{self._bar(bar, fraction)}] {fraction * 100:5.1f}% | "
            f"{self._status(status_label(status_value), status_value):<18} | phase {phase:<8} "
            f"| selected {selected}, submitted {submitted}, caseRuns {_case_run_count_label(case_runs, visible_case_runs)}, "
            f"failed {failed} "
            f"| cases {_case_summary_label(payload)} | metrics {metric} | elapsed {elapsed} | stale {format_duration(stable_for)} "
            f"| evalRunId {eval_run_id}"
        )

    def _render_panel(self, payload: dict[str, Any], diagnostics: dict[str, Any], stable_for: float) -> None:
        self._clear_panel()
        lines = self._panel_lines(payload, diagnostics, stable_for)
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self.panel_lines = len(lines)

    def _clear_panel(self) -> None:
        if self.panel_lines <= 0:
            return
        self.stream.write(f"\x1b[{self.panel_lines}F")
        self.stream.write("\x1b[J")
        self.stream.flush()
        self.panel_lines = 0

    def _panel_lines(self, payload: dict[str, Any], diagnostics: dict[str, Any], stable_for: float) -> list[str]:
        width = _terminal_width()
        fraction = progress_fraction(payload, self.selected_cases)
        status_value = str(payload.get("status") or "UNKNOWN")
        selected = _int(payload.get("selectedCaseCount")) or self.selected_cases
        submitted = _int(payload.get("submittedCaseCount"))
        case_runs = _int(payload.get("caseRunCount"))
        visible_case_runs = _known_case_count(payload)
        failed = _int(payload.get("failedCaseCount"))
        eval_run_id = str(payload.get("evalRunId") or "-")
        phase = str(payload.get("phase") or "-")
        message = str(payload.get("message") or payload.get("errorMessage") or "-").strip()
        elapsed = format_duration(time.monotonic() - self.started_at)
        lines = [
            self._rule(width),
            self._title("AgentX Bench Run Monitor"),
            self._rule(width),
            self._section("Run"),
            f"Status   : {self._status(status_label(status_value), status_value)} ({status_value})",
            f"Progress : [{self._bar(progress_bar(fraction, 36), fraction)}] {fraction * 100:5.1f}%"
            f"    elapsed {elapsed}    stale {self._stale(stable_for)}",
            f"Stages   : {self._stages(stage_line(status_value))}",
            f"Cases    : selected {selected} | submitted {submitted} | "
            f"caseRuns {_case_run_count_label(case_runs, visible_case_runs)} | failed {failed}"
            f" | {_case_summary_label(payload)}",
            f"Metrics  : {_metric_label(payload)}",
            f"EvalRun  : {eval_run_id}",
            f"Phase    : {phase}",
        ]
        lines.extend(_wrapped("Now      : ", stage_guidance(status_value), width))
        lines.extend(_wrapped("Message  : ", message, width))
        case_lines = self._case_progress_lines(payload, compact=False, width=width)
        if case_lines:
            lines.append(self._section("Case Progress"))
            lines.extend(case_lines)
        if payload.get("caseProgressError"):
            lines.extend(_wrapped("Case API : ", self._warn(str(payload.get("caseProgressError"))), width))
        warnings = diagnostics.get("warnings") or []
        if warnings:
            lines.append(self._section("Warnings"))
            for warning in warnings[:4]:
                lines.extend(_wrapped("  - ", self._warn(warning), width))
        next_actions = diagnostics.get("nextActions") or []
        if next_actions:
            lines.append(self._section("Next"))
            for action in next_actions[:3]:
                lines.extend(_wrapped("  - ", action, width))
        if self.run_dir:
            lines.append(self._section("Artifacts"))
            lines.extend(_wrapped("Artifacts: ", str(self.run_dir), width))
        lines.append(self._rule(width))
        return lines

    def _case_progress_lines(self, payload: dict[str, Any], compact: bool,
                             width: int | None = None) -> list[str]:
        rows = payload.get("caseProgress") or []
        if not rows:
            return []
        width = width or _terminal_width()
        limit = 6 if compact else 10
        lines: list[str] = []
        if compact:
            lines.append(f"  cases   : {_case_summary_label(payload)}")
        header = f"  {'#':>2}  {'caseId':<32} {'status':<12} {'stage':<14} {'caseRunId':<18} note"
        lines.append(self._muted(header))
        for row in rows[:limit]:
            status = str(row.get("status") or "-")
            status_text = f"{_clip(status, 12):<12}"
            note = str(row.get("message") or "-")
            line = (
                f"  {_int(row.get('index')):>2}  "
                f"{_clip(str(row.get('caseId') or '-'), 32):<32} "
                f"{self._status(status_text, status)} "
                f"{_clip(str(row.get('stage') or '-'), 14):<14} "
                f"{_clip(str(row.get('caseRunId') or '-'), 18):<18} "
                f"{_clip(note, max(16, width - 86))}"
            )
            lines.append(line)
        remaining = len(rows) - limit
        if remaining > 0:
            lines.append(self._muted(f"  ... {remaining} more case(s); full detail is written into status_history.jsonl"))
        return lines

    def _should_emit_waiting_warning(self, stable_for: float) -> bool:
        if stable_for < 120:
            return False
        bucket = int(stable_for // 120)
        if bucket == self.last_warning_bucket:
            return False
        self.last_warning_bucket = bucket
        return True

    def _resolve_interactive(self) -> bool:
        if self.mode == "panel":
            return True
        if self.mode == "lines":
            return False
        if self.mode != "auto":
            raise ValueError(f"unsupported progress mode: {self.mode}")
        isatty = getattr(self.stream, "isatty", None)
        return bool(isatty and isatty())

    def _resolve_prompt_enabled(self) -> bool:
        if self.mode == "lines":
            return False
        output_isatty = getattr(self.stream, "isatty", None)
        input_isatty = getattr(self.input_stream, "isatty", None)
        return bool(output_isatty and output_isatty() and input_isatty and input_isatty())

    def _resolve_color(self) -> bool:
        if self.color == "always":
            return True
        if self.color == "never":
            return False
        if self.color != "auto":
            raise ValueError(f"unsupported color mode: {self.color}")
        if os.environ.get("NO_COLOR"):
            return False
        isatty = getattr(self.stream, "isatty", None)
        return bool(isatty and isatty())

    def _write_start_banner(self, context: RunContext, selected_cases: int, method: str, run_dir: Path,
                            service_url: str) -> None:
        lines = [
            "",
            *AGENTX_BANNER,
            " " * AGENTX_BANNER_WIDTH,
            f"Agent Evaluation Runner v{__version__}".center(AGENTX_BANNER_WIDTH),
            "",
            "▰" * AGENTX_BANNER_WIDTH,
            "",
            f" ✓ Environment : {_environment_label()}",
            f" ✓ Proxy       : {_proxy_label(method, service_url)}",
            f" ✓ Framework   : {_framework_label(method)}",
            f" ✓ Runtime     : {_runtime_label(self.timeout_seconds, self.poll_interval_seconds)}",
            f" ✓ Evaluator   : {_evaluator_label(context, selected_cases)}",
            "",
            f" [❖] Targets : {_target_label(context)}",
            f" [❖] Models  : {_model_label(context)}",
            f" [❖] Suite   : {context.suite.suite_id} ({selected_cases}/{len(context.suite.cases)} cases)",
            f" [❖] Output  : {run_dir}",
            "",
            " ✓ Ready      : Prepared for review",
            "",
        ]
        banner_start = 1
        banner_end = banner_start + len(AGENTX_BANNER)
        for index, line in enumerate(lines):
            if banner_start <= index < banner_end:
                self._write(self._title(line))
            elif line.startswith("▰"):
                self._write(self._bar(line, 0.72))
            elif line.startswith(" ✓ Ready"):
                self._write(self._status(line, "LOCAL_PREPARING"))
            else:
                self._write(line)

    def _write_case_plan(self, cases: list[BenchmarkCase]) -> None:
        width = _terminal_width()
        self._write(self._section("Planned Cases"))
        header = f"  {'#':>2}  {'caseId':<34} {'caseType':<18} {'category':<18} {'difficulty':<10} name"
        self._write(self._muted(header))
        name_width = max(18, width - 93)
        for index, case in enumerate(cases, start=1):
            attributes = case.attributes_json or {}
            line = (
                f"  {index:>2}  "
                f"{_clip(case.case_id, 34):<34} "
                f"{_clip(str(attributes.get('caseType') or '-'), 18):<18} "
                f"{_clip(str(attributes.get('category') or '-'), 18):<18} "
                f"{_clip(str(attributes.get('difficulty') or '-'), 10):<10} "
                f"{_clip(case.name, name_width)}"
            )
            self._write(line)
        self._write("")

    def _write_start_hint(self) -> None:
        if not self.enabled:
            return
        if self.prompt_enabled:
            self._write("Review the planned cases above before starting.")
        else:
            self._write("Non-interactive shell detected; evaluation will start automatically.")
        self._write("")

    def _write(self, line: str) -> None:
        if not self.enabled:
            return
        self.stream.write(line + "\n")
        self.stream.flush()

    def _rule(self, width: int) -> str:
        return self._muted("-" * width)

    def _title(self, text: str) -> str:
        return self._color(text, "cyan", bold=True)

    def _section(self, text: str) -> str:
        return self._color(f"[{text}]", "blue", bold=True)

    def _status(self, text: str, status: str) -> str:
        return self._color(text, _status_color(status), bold=True)

    def _bar(self, text: str, fraction: float) -> str:
        if fraction >= 1:
            color = "green"
        elif fraction >= 0.8:
            color = "cyan"
        elif fraction >= 0.5:
            color = "blue"
        else:
            color = "yellow"
        return self._color(text, color)

    def _warn(self, text: str) -> str:
        return self._color(text, "yellow")

    def _muted(self, text: str) -> str:
        return self._color(text, "muted")

    def _stale(self, seconds: float) -> str:
        text = format_duration(seconds)
        if seconds >= 300:
            return self._color(text, "red", bold=True)
        if seconds >= 120:
            return self._color(text, "yellow", bold=True)
        return text

    def _stages(self, text: str) -> str:
        if not self.color_enabled:
            return text
        return (
            text.replace(" OK", f" {ANSI['green']}OK{ANSI['reset']}")
            .replace(" >>", f" {ANSI['cyan']}>>{ANSI['reset']}")
            .replace(" ..", f" {ANSI['muted']}..{ANSI['reset']}")
        )

    def _color(self, text: str, color: str, bold: bool = False) -> str:
        if not self.color_enabled:
            return text
        prefix = ANSI.get(color, "")
        if bold:
            prefix = ANSI["bold"] + prefix
        return f"{prefix}{text}{ANSI['reset']}"


def _environment_label() -> str:
    system = platform.system() or "Unknown"
    if system == "Darwin":
        name = "macOS (Darwin)"
    elif system == "Linux":
        name = "Linux"
    elif system == "Windows":
        name = "Windows"
    else:
        name = system
    machine = platform.machine() or "local"
    return f"{name} {machine} Workspace"


def _proxy_label(method: str, service_url: str) -> str:
    normalized_method = str(method or "").lower()
    if normalized_method in {"energent_swarm", "energent_swarm_service"}:
        if service_url:
            return f"AgentX API {service_url} initialized"
        return "AgentX API service runner initialized"
    return f"{method or 'local'} runner initialized"


def _framework_label(method: str) -> str:
    normalized_method = str(method or "").lower()
    if normalized_method in {"energent_swarm", "energent_swarm_service"}:
        return "Energent Core / SOFA-Boot loaded"
    return "AgentX Bench local runtime loaded"


def _runtime_label(timeout_seconds: float | None, poll_interval_seconds: float | None) -> str:
    parts = []
    if timeout_seconds is None:
        parts.append("timeout default")
    elif timeout_seconds <= 0:
        parts.append("timeout immediate")
    else:
        parts.append(f"timeout {format_duration(timeout_seconds)}")
    if poll_interval_seconds is not None and poll_interval_seconds > 0:
        parts.append(f"poll {format_duration(poll_interval_seconds)}")
    return " · ".join(parts)


def _evaluator_label(context: RunContext, selected_cases: int) -> str:
    metric_count = len(context.metric_codes)
    return f"{context.suite.suite_id} · {selected_cases} cases · {metric_count} metrics"


def _target_label(context: RunContext) -> str:
    subject = context.subject
    return f"{subject.subject_name} [{subject.subject_id}] · team={subject.team_name}"


def _model_label(context: RunContext) -> str:
    subject = context.subject
    parts = [subject.agent_mode or "unknown-mode", subject.model or "model not specified"]
    if subject.version:
        parts.append(f"version={subject.version}")
    return " / ".join(parts)


def progress_fraction(payload: dict[str, Any], selected_cases: int = 0) -> float:
    status_value = str(payload.get("status") or "").upper()
    selected = _int(payload.get("selectedCaseCount")) or selected_cases
    submitted = _int(payload.get("submittedCaseCount"))
    case_runs = max(_int(payload.get("caseRunCount")), _known_case_count(payload))

    if status_value in {"COMPLETED", "COMPLETED_WITH_SKIPPED_METRICS", "RESULTS_READY"}:
        return 1.0
    if status_value in {"FAILED", "VALIDATION_FAILED", "UNKNOWN_OR_EXPIRED", "TIMEOUT", "INCOMPLETE",
                        "INTERRUPTED", "CANCELED"}:
        return 1.0
    if status_value == "LOCAL_PREPARING":
        return STAGE_PROGRESS["LOCAL_PREPARING"]
    if status_value == "FETCHING_RESULTS":
        return STAGE_PROGRESS["FETCHING_RESULTS"]
    if status_value == "SUBMITTING_CASES":
        return STAGE_PROGRESS["SUBMITTING_CASES"] + 0.10 * _case_ratio(submitted, selected)
    if status_value == "COLLECTING_TRACE":
        return STAGE_PROGRESS["COLLECTING_TRACE"] + 0.10 * _case_ratio(case_runs, selected)
    if status_value == "COMPUTING_METRICS":
        return STAGE_PROGRESS["COMPUTING_METRICS"] + (0.10 if payload.get("metricComputed") else 0.0)
    if status_value == "SUBMITTED":
        return STAGE_PROGRESS["SUBMITTED"]
    if status_value == "RUNNING":
        return STAGE_PROGRESS["RUNNING"]
    return min(0.95, 0.12 + 0.50 * _case_ratio(max(submitted, case_runs), selected))


def progress_bar(fraction: float, width: int = 24) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


def build_dashboard(summary: dict[str, Any], run_dir: Path) -> str:
    diagnostics = summary.get("diagnostics") or {}
    latest_status = summary.get("latestStatus") or diagnostics.get("latestStatus") or {}
    dimension_scores = summary.get("dimensionScores") or {}
    metric_summary = summary.get("metricSummary") or {}
    evidence_summary = summary.get("evidenceSummary") or {}
    lines = [
        "AgentX Bench Dashboard",
        "=" * 72,
        f"Run        : {summary.get('runId') or '-'}",
        f"EvalRun    : {summary.get('evalRunId') or '-'}",
        f"Subject    : {summary.get('subjectName') or summary.get('subjectId') or '-'}",
        f"Status     : {status_label(summary.get('status'))} ({summary.get('status') or '-'})",
        f"Artifacts  : {run_dir}",
        "",
        "Scores",
        f"  total          : {_fmt_score(summary.get('totalScore'))}",
        f"  outcome        : {_fmt_score(summary.get('outcomeScore'))}",
        f"  execution      : {_fmt_score(summary.get('executionScore'))}",
        f"  orchestration  : {_fmt_score(summary.get('orchestrationScore'))}",
        f"  efficiency     : {_fmt_score(summary.get('efficiencyScore'))}",
        "",
        "Run Stats",
        f"  cases          : {summary.get('completedCases', 0)}/{summary.get('selectedCases', 0)} completed",
        f"  failed         : {summary.get('failedCases', 0)} unscorable"
        f" | runtime {summary.get('runtimeFailedCases', summary.get('failedCases', 0))}",
        f"  scorable failed: {summary.get('scorableNonSuccessCases', 0)}",
        f"  success rate   : {_fmt_percent(summary.get('successRate'))}",
        f"  avg duration   : {_fmt_duration_ms(summary.get('avgDurationMs'))}",
        f"  token usage    : {_fmt_number(summary.get('tokenUsage'))}",
    ]
    if dimension_scores:
        lines.extend(["", "Dimension Breakdown"])
        for dimension in ("outcome", "execution", "orchestration", "efficiency"):
            if dimension in dimension_scores:
                lines.append(
                    f"  {dimension:<14}: {_score_bar(dimension_scores.get(dimension))} "
                    f"{_fmt_score(dimension_scores.get(dimension))}"
                )
    if metric_summary:
        lines.extend(["", "Score Coverage"])
        lines.extend(_metric_summary_lines(metric_summary))
    if evidence_summary:
        lines.extend(["", "Evidence Coverage"])
        lines.extend(_evidence_summary_lines(evidence_summary))
    if latest_status or diagnostics:
        lines.extend([
            "",
            "Run Diagnostics",
            f"  latest status  : {status_label(latest_status.get('status'))} ({latest_status.get('status') or '-'})",
            f"  phase          : {latest_status.get('phase') or '-'}",
            f"  message        : {latest_status.get('message') or latest_status.get('errorMessage') or '-'}",
            f"  selected       : {latest_status.get('selectedCaseCount', summary.get('selectedCases', 0))}",
            f"  submitted      : {latest_status.get('submittedCaseCount', '-')}",
            f"  caseRuns       : {latest_status.get('caseRunCount', '-')}",
            f"  metrics        : {_metric_label(latest_status)}",
        ])
        warnings = diagnostics.get("warnings") or []
        if warnings:
            lines.append("  warnings       :")
            for warning in warnings[:6]:
                lines.append(f"    - {warning}")
        next_actions = diagnostics.get("nextActions") or []
        if next_actions:
            lines.append("  next actions   :")
            for action in next_actions[:6]:
                lines.append(f"    - {action}")
    lines.extend(["", "Data Quality"])
    lines.extend(_data_quality_lines(summary))
    artifact_lines = _artifact_lines(run_dir)
    if artifact_lines:
        lines.extend(["", "Artifacts"])
        lines.extend(artifact_lines)
    focus_cases = _focus_cases(summary.get("caseScores") or [])
    if focus_cases:
        lines.extend(["", "Attention Cases"])
        for row in focus_cases[:8]:
            lines.append(
                f"  - {row.get('caseId')} | {row.get('status')} | score {_fmt_score(row.get('caseScore'))} "
                f"| {row.get('caseName')}"
            )
            reasons = row.get("reasons") or []
            if reasons:
                lines.extend(_wrapped("    reason: ", "; ".join(str(item) for item in reasons[:2]), 72))
    else:
        lines.extend(["", "Attention Cases", "  none"])
    return "\n".join(lines) + "\n"


def build_run_diagnostics(payload: dict[str, Any] | None, selected_cases: int = 0,
                          stable_for_seconds: float | None = None,
                          error_message: str | None = None) -> dict[str, Any]:
    payload = dict(payload or {})
    status_value = str(payload.get("status") or "UNKNOWN").upper()
    selected = _int(payload.get("selectedCaseCount")) or selected_cases
    submitted = _int(payload.get("submittedCaseCount"))
    case_runs = _int(payload.get("caseRunCount"))
    visible_case_runs = _known_case_count(payload)
    failed = _int(payload.get("failedCaseCount"))
    metric_computed = payload.get("metricComputed")
    warnings: list[str] = []
    next_actions: list[str] = []

    if error_message:
        warnings.append(f"CLI stopped before a complete benchmark result was available: {error_message}")
    if status_value == "COLLECTING_TRACE" and visible_case_runs == 0:
        warnings.append("Trace collection is active but the service has not reported any caseRunIds yet.")
        next_actions.append("Check the evaluation service logs around submitPlanAndCollectTrace and case submission.")
    case_summary = payload.get("caseProgressSummary") or {}
    if status_value in {"COLLECTING_TRACE", "COMPUTING_METRICS"} and _int(case_summary.get("waiting")) == selected:
        warnings.append("All selected cases are still waiting; no case-level durable progress is visible yet.")
    if visible_case_runs > case_runs:
        warnings.append(
            f"Service status counters look stale; case detail endpoint shows {visible_case_runs} visible caseRun(s)."
        )
    if status_value in {"SUBMITTING_CASES", "COLLECTING_TRACE", "COMPUTING_METRICS"} \
            and selected and submitted == 0 and visible_case_runs == 0:
        warnings.append("No submitted cases are visible in the service status, so the run may still be before durable case execution.")
        next_actions.append("Verify that the selected suite, teamName, and orchestration host are valid.")
    if status_value == "COMPUTING_METRICS" and visible_case_runs == 0:
        warnings.append("Metric computation is active but no caseRun evidence is visible.")
    if metric_computed is False and status_value == "COMPLETED_WITH_SKIPPED_METRICS":
        warnings.append("Metrics are not fully computed; score-bearing quality metrics may be missing.")
    if failed:
        warnings.append(f"{failed} case(s) have failed according to the service status.")
    if stable_for_seconds is not None and stable_for_seconds >= 120 and status_value not in TERMINAL_DISPLAY_STATUSES:
        warnings.append(f"No backend progress has been observed for {format_duration(stable_for_seconds)}.")
        next_actions.append("Keep waiting only if the host service is healthy; otherwise inspect service logs and the evalRunId.")
    if status_value == "TIMEOUT":
        next_actions.append("Increase --timeout only after confirming the service is still making progress.")
    if status_value == "FAILED":
        next_actions.append("Use errors.jsonl, run_status.json, and service logs to identify the failing stage.")
    if not payload.get("evalRunId") and status_value in {
        "SUBMITTED",
        "RUNNING",
        "SUBMITTING_CASES",
        "COLLECTING_TRACE",
        "COMPUTING_METRICS",
        "FETCHING_RESULTS",
        "RESULTS_READY",
    }:
        warnings.append("No evalRunId is available, so the service may not have accepted the run.")

    return {
        "latestStatus": payload,
        "status": status_value,
        "selectedCaseCount": selected,
        "submittedCaseCount": submitted,
        "caseRunCount": case_runs,
        "failedCaseCount": failed,
        "stableForSeconds": round(stable_for_seconds, 2) if stable_for_seconds is not None else None,
        "warnings": _unique(warnings),
        "nextActions": _unique(next_actions),
    }


def progress_signature(payload: dict[str, Any]) -> str:
    keys = (
        "status",
        "phase",
        "selectedCaseCount",
        "submittedCaseCount",
        "caseRunCount",
        "failedCaseCount",
        "metricComputed",
        "metricMessage",
        "updatedAt",
        "completedAt",
        "errorCode",
        "errorMessage",
        "message",
    )
    values = [str(payload.get(key) or "") for key in keys]
    values.append(",".join(str(item) for item in payload.get("caseRunIds") or []))
    values.append(",".join(str(item) for item in payload.get("failedCaseIds") or []))
    for row in payload.get("caseProgress") or []:
        values.append("|".join([
            str(row.get("caseId") or ""),
            str(row.get("caseRunId") or ""),
            str(row.get("status") or ""),
            str(row.get("stage") or ""),
            str(row.get("message") or ""),
        ]))
    return "\x1f".join(values)


def status_label(value: Any) -> str:
    status_value = str(value or "-").upper()
    labels = {
        "LOCAL_PREPARING": "Preparing",
        "SUBMITTED": "Accepted",
        "RUNNING": "Running",
        "SUBMITTING_CASES": "Submitting cases",
        "COLLECTING_TRACE": "Collecting trace",
        "COMPUTING_METRICS": "Computing metrics",
        "FETCHING_RESULTS": "Fetching results",
        "RESULTS_READY": "Results ready",
        "COMPLETED": "Completed",
        "COMPLETED_WITH_SKIPPED_METRICS": "Completed with skipped metrics",
        "CANCELED": "Canceled",
        "FAILED": "Failed",
        "VALIDATION_FAILED": "Validation failed",
        "UNKNOWN_OR_EXPIRED": "Unknown or expired",
        "TIMEOUT": "Timed out",
        "INCOMPLETE": "Incomplete",
        "INTERRUPTED": "Interrupted",
    }
    return labels.get(status_value, status_value)


def stage_guidance(status: str) -> str:
    status_value = str(status or "").upper()
    if status_value in TERMINAL_DISPLAY_STATUSES:
        return "The run reached a terminal state; inspect the completion summary, diagnostics, and artifacts."
    return STAGE_GUIDANCE.get(status_value, "Waiting for the next Evaluation Service status update.")


def stage_line(status: str) -> str:
    status_value = str(status or "").upper()
    if status_value in {"COMPLETED", "COMPLETED_WITH_SKIPPED_METRICS", "RESULTS_READY"}:
        current_index = len(STAGES)
    elif status_value in {"FAILED", "VALIDATION_FAILED", "UNKNOWN_OR_EXPIRED", "TIMEOUT", "INCOMPLETE",
                          "INTERRUPTED", "CANCELED"}:
        current_index = STAGE_INDEX.get(status_value, -1)
    else:
        current_index = STAGE_INDEX.get(status_value, -1)
    labels = []
    for index, (stage_status, label) in enumerate(STAGES):
        if current_index >= len(STAGES) or index < current_index:
            marker = "OK"
        elif index == current_index:
            marker = ">>"
        else:
            marker = ".."
        labels.append(f"{label} {marker}")
    return " | ".join(labels)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _focus_cases(case_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = [row for row in case_scores if str(row.get("status") or "").upper() not in {"SUCCESS", "COMPLETED"}]
    if failed:
        return failed
    return sorted(case_scores, key=lambda row: float(row.get("caseScore") or 0.0))[:5]


def _data_quality_lines(summary: dict[str, Any]) -> list[str]:
    selected = _int(summary.get("selectedCases"))
    completed = _int(summary.get("completedCases"))
    failed = _int(summary.get("failedCases"))
    incomplete = _int(summary.get("incompleteCases"))
    not_completed = max(selected - completed, failed, incomplete, 0)
    status_value = str(summary.get("status") or "").upper()
    completeness = summary.get("resultCompleteness") or {}
    diagnostics = summary.get("diagnostics") or {}
    warnings = diagnostics.get("warnings") or []
    metric_summary = summary.get("metricSummary") or {}
    score_confidence = str(metric_summary.get("scoreConfidence") or "unknown")
    eligible = _leaderboard_eligible(summary)
    runtime_failed = _int(summary.get("runtimeFailedCases"))
    scorable_non_success = _int(summary.get("scorableNonSuccessCases"))
    unscorable_failed = _int(summary.get("unscorableFailedCases"))
    lines = [
        f"  leaderboard    : {'eligible' if eligible else 'diagnostic only'}",
        f"  score confidence: {score_confidence}",
        f"  case coverage  : {completed}/{selected} completed, {not_completed} not completed",
        f"  case counters  : failedCases={failed}, incompleteCases={incomplete}, runtimeFailedCases={runtime_failed}, "
        f"scorableNonSuccessCases={scorable_non_success}, unscorableFailedCases={unscorable_failed}",
        f"  warning count  : {len(warnings)}",
    ]
    if status_value not in {"COMPLETED", "SUCCESS"}:
        lines.append(f"  exclusion      : terminal status {status_value or 'UNKNOWN'} is not complete")
    elif failed or incomplete:
        lines.append("  exclusion      : unscorable failed, timed-out, or incomplete cases exist")
    elif score_confidence == "diagnostic":
        lines.append("  exclusion      : quality judge metrics are unavailable")
    elif not eligible:
        lines.append("  exclusion      : data quality does not meet leaderboard criteria")
    else:
        lines.append("  exclusion      : none")
    if completeness:
        lines.append("  completeness   :")
        lines.extend(f"    - {item}" for item in _fmt_completeness(completeness))
    return lines


def _leaderboard_eligible(summary: dict[str, Any]) -> bool:
    status_value = str(summary.get("status") or "").upper()
    if status_value not in {"COMPLETED", "SUCCESS"}:
        return False
    if _int(summary.get("incompleteCases")) or _int(summary.get("failedCases")):
        return False
    metric_summary = summary.get("metricSummary") or {}
    if metric_summary and metric_summary.get("scoreConfidence") == "diagnostic":
        return False
    return summary.get("totalScore") is not None


def _metric_summary_lines(metric_summary: dict[str, Any]) -> list[str]:
    quality_expected = _int(metric_summary.get("qualityExpected"))
    quality_scored = _int(metric_summary.get("qualityScored"))
    quality_unavailable = _int(metric_summary.get("qualityUnavailable"))
    quality_missing = _int(metric_summary.get("qualityMissing"))
    quality_not_applicable = _int(metric_summary.get("qualityNotApplicable"))
    derived_expected = _int(metric_summary.get("derivedExpected"))
    derived_available = _int(metric_summary.get("derivedAvailable"))
    confidence = str(metric_summary.get("scoreConfidence") or "unknown")
    outcomes = metric_summary.get("outcomes") or {}
    lines = [
        f"  confidence     : {confidence}",
        f"  quality metrics: {quality_scored}/{quality_expected} scored"
        f" | unavailable {quality_unavailable} | missing {quality_missing} | n/a {quality_not_applicable}",
        f"  efficiency data: {derived_available}/{derived_expected} available",
        f"  metric results : {_int(metric_summary.get('totalMetricResults'))}",
    ]
    if outcomes:
        ordered = ["scored", "statistic", "unavailable", "not_applicable", "failed", "skipped"]
        parts = [f"{key}={_int(outcomes.get(key))}" for key in ordered if _int(outcomes.get(key))]
        lines.append(f"  outcomes       : {', '.join(parts) if parts else '-'}")
    by_metric = metric_summary.get("byMetric") or {}
    if by_metric:
        lines.append("  metric detail  :")
        for metric_code, counts in list(by_metric.items())[:8]:
            parts = [f"{key}:{value}" for key, value in counts.items()]
            lines.append(f"    - {metric_code}: {', '.join(parts)}")
    if confidence == "diagnostic":
        lines.append("  note           : quality judge metrics are not usable enough for an official leaderboard score")
    elif confidence == "limited":
        lines.append("  note           : some quality metrics are unavailable or missing; treat ranking with caution")
    return lines


def _evidence_summary_lines(evidence_summary: dict[str, Any]) -> list[str]:
    return [
        f"  case results   : {_int(evidence_summary.get('caseResultCount'))}",
        f"  final answers  : {_int(evidence_summary.get('finalAnswerCount'))}",
        f"  events         : {_int(evidence_summary.get('eventCount'))}",
        f"  raw traces     : {_int(evidence_summary.get('rawTraceCount'))}",
        f"  trace summaries: success {_int(evidence_summary.get('traceSuccessCount'))}"
        f" | failed {_int(evidence_summary.get('traceFailedCount'))}",
    ]


def _fmt_completeness(value: dict[str, Any]) -> list[str]:
    parts = []
    for key in sorted(value.keys()):
        item = value.get(key)
        if isinstance(item, dict):
            for nested_key, nested_value in sorted(item.items()):
                parts.append(f"{key}.{nested_key}={nested_value}")
        else:
            parts.append(f"{key}={item}")
    return parts[:12]


def _artifact_lines(run_dir: Path) -> list[str]:
    names = [
        "request.json",
        "run_status.json",
        "status_history.jsonl",
        "raw_results.json",
        "summary.json",
        "case_results.csv",
        "case_results.jsonl",
        "leaderboard_row.csv",
        "dashboard.txt",
        "diagnostic_report.txt",
        "errors.jsonl",
    ]
    if not run_dir.exists():
        return []
    generated_by_this_report = {"dashboard.txt", "diagnostic_report.txt"}
    lines = []
    for name in names:
        if (run_dir / name).exists() or name in generated_by_this_report:
            state = "ok"
        else:
            state = "missing"
        lines.append(f"  {name:<22}: {state}")
    return lines


def _score_bar(value: Any, width: int = 20) -> str:
    number = _float(value)
    if number is None:
        return "[" + "." * width + "]"
    fraction = max(0.0, min(1.0, number / 100.0))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _case_ratio(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, value / total))


def _case_summary_label(payload: dict[str, Any]) -> str:
    summary = payload.get("caseProgressSummary") or {}
    if not summary:
        return "detail pending"
    parts = []
    for key, label in (
        ("waiting", "waiting"),
        ("running", "running"),
        ("success", "success"),
        ("failed", "failed"),
        ("incomplete", "incomplete"),
    ):
        count = _int(summary.get(key))
        if count:
            parts.append(f"{label} {count}")
    return ", ".join(parts) if parts else "detail pending"


def _known_case_count(payload: dict[str, Any]) -> int:
    rows = payload.get("caseProgress") or []
    return sum(
        1
        for row in rows
        if str(row.get("caseRunId") or "").strip()
        or str(row.get("status") or "").upper() in {"SUCCESS", "COMPLETED", "FAILED", "TIMEOUT", "ERROR"}
    )


def _case_run_count_label(status_count: int, visible_count: int) -> str:
    if visible_count > status_count:
        return f"{status_count} (visible {visible_count})"
    return str(status_count)


def _metric_label(payload: dict[str, Any]) -> str:
    if not payload:
        return "-"
    if payload.get("metricComputed") is True:
        return "done"
    if payload.get("metricComputed") is False:
        return "pending"
    if payload.get("metricMessage"):
        return str(payload["metricMessage"])
    return "-"


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _fmt_score(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    return f"{number:.2f}"


def _fmt_percent(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"


def _fmt_duration_ms(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    if number >= 1000:
        return f"{number / 1000:.2f}s"
    return f"{number:.0f}ms"


def _fmt_number(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    return f"{number:.2f}"


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _terminal_width() -> int:
    return max(88, min(118, shutil.get_terminal_size(fallback=(100, 24)).columns))


def _clip(value: str, width: int) -> str:
    if width <= 1:
        return value[:width]
    if len(value) <= width:
        return value
    return value[:width - 1] + "~"


def _wrapped(prefix: str, text: str, width: int) -> list[str]:
    text = text or "-"
    available = max(20, width - len(prefix))
    chunks = textwrap.wrap(text, width=available) or ["-"]
    lines = [prefix + chunks[0]]
    continuation = " " * len(prefix)
    lines.extend(continuation + chunk for chunk in chunks[1:])
    return lines


def _status_color(status: str) -> str:
    status_value = str(status or "").upper()
    if status_value in {"SUCCESS", "COMPLETED"}:
        return "green"
    if status_value in {"FAILED", "TIMEOUT", "ERROR", "VALIDATION_FAILED"}:
        return "red"
    if status_value in {"COMPLETED_WITH_SKIPPED_METRICS", "UNAVAILABLE", "INCOMPLETE"}:
        return "yellow"
    if status_value in {"RUNNING", "SUBMITTING_CASES", "COLLECTING_TRACE", "COMPUTING_METRICS"}:
        return "cyan"
    if status_value in {"WAITING", "PENDING", "QUEUED", "SUBMITTED"}:
        return "blue"
    return "muted"


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
