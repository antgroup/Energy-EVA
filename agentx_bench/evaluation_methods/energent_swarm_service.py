"""Energent Swarm adapter backed by the AgentX Evaluation Service API."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from ..evaluation_utils.artifacts import ArtifactWriter
from ..evaluation_utils.client import AgentXClient, AgentXServiceError
from ..evaluation_utils.models import BenchmarkCase, RunContext, TERMINAL_STATUSES


class EnergentSwarmServiceRunner:
    def __init__(self, client: AgentXClient, timeout_seconds: float, poll_interval_seconds: float,
                 include_events: bool = False, include_metric_results: bool = True,
                 include_raw_trace: bool = False, sleep_fn: Callable[[float], None] = time.sleep,
                 progress_callback: Callable[[dict[str, Any]], None] | None = None):
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.include_events = include_events
        self.include_metric_results = include_metric_results
        self.include_raw_trace = include_raw_trace
        self.sleep_fn = sleep_fn
        self.progress_callback = progress_callback

    def run(self, context: RunContext, rendered_cases: list[BenchmarkCase],
            writer: ArtifactWriter) -> dict[str, Any]:
        payload = build_submit_payload(context, rendered_cases)
        writer.write_json("request.json", payload)
        self._publish_status(writer, enrich_progress_status(
            context,
            rendered_cases,
            _local_status(context, "LOCAL_PREPARING", "request rendered"),
        ))

        accepted = self.client.submit_run(payload)
        eval_run_id = str(accepted.get("evalRunId") or "")
        if not eval_run_id:
            raise RuntimeError("AgentX service accepted response missing evalRunId")
        self._publish_status(writer, enrich_progress_status(context, rendered_cases, {**accepted, "observedAt": _now()}))

        deadline = time.monotonic() + self.timeout_seconds
        latest_status = accepted
        while True:
            latest_status = self.client.get_status(eval_run_id)
            progress_results, progress_error = self._progress_results(eval_run_id)
            progress_status = enrich_progress_status(
                context,
                rendered_cases,
                {**latest_status, "observedAt": _now()},
                progress_results,
                progress_error,
            )
            self._publish_status(writer, progress_status)
            status_value = str(latest_status.get("status") or "")
            if status_value in TERMINAL_STATUSES:
                break
            if time.monotonic() >= deadline:
                timeout_status = {
                    **latest_status,
                    "status": "TIMEOUT",
                    "message": f"timed out after {self.timeout_seconds} seconds",
                    "observedAt": _now(),
                }
                self._publish_status(writer, enrich_progress_status(
                    context,
                    rendered_cases,
                    timeout_status,
                    progress_results,
                    progress_error,
                ))
                raise TimeoutError(f"AgentX run timed out, evalRunId={eval_run_id}, lastStatus={status_value}")
            self.sleep_fn(self.poll_interval_seconds)

        self._notify_progress(enrich_progress_status(context, rendered_cases, {
            **latest_status,
            "status": "FETCHING_RESULTS",
            "phase": "download_results",
            "message": "downloading run results",
            "observedAt": _now(),
        }, progress_results))
        results = self.client.get_results(
            eval_run_id,
            include_events=self.include_events,
            include_metric_results=self.include_metric_results,
            include_raw_trace=self.include_raw_trace,
        )
        normalized = normalize_service_results(context, rendered_cases, latest_status, results)
        writer.write_json("raw_results.json", normalized)
        self._notify_progress(enrich_progress_status(context, rendered_cases, {
            **latest_status,
            "status": "RESULTS_READY",
            "phase": "score_locally",
            "message": "results downloaded",
            "observedAt": _now(),
        }, results))
        return normalized

    def _progress_results(self, eval_run_id: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return self.client.get_results(
                eval_run_id,
                include_events=False,
                include_metric_results=False,
                include_raw_trace=False,
            ), None
        except AgentXServiceError as exc:
            return None, str(exc)
        except RuntimeError as exc:
            return None, str(exc)

    def _publish_status(self, writer: ArtifactWriter, payload: dict[str, Any]) -> None:
        writer.append_status(payload)
        self._notify_progress(payload)

    def _notify_progress(self, payload: dict[str, Any]) -> None:
        if self.progress_callback:
            self.progress_callback(payload)


def build_submit_payload(context: RunContext, rendered_cases: list[BenchmarkCase]) -> dict[str, Any]:
    return {
        "clientRunId": context.run_id,
        "suite": context.suite.suite_payload(),
        "subject": context.subject.to_payload(),
        "metricCodes": context.metric_codes,
        "cases": [case.to_payload() for case in rendered_cases],
    }


def enrich_progress_status(context: RunContext, rendered_cases: list[BenchmarkCase], status: dict[str, Any],
                           results: dict[str, Any] | None = None,
                           progress_error: str | None = None) -> dict[str, Any]:
    payload = dict(status)
    case_progress = build_case_progress(rendered_cases, payload, results)
    payload["caseProgress"] = case_progress
    payload["caseProgressSummary"] = _case_progress_summary(case_progress)
    if progress_error:
        payload["caseProgressError"] = progress_error
    payload.setdefault("selectedCaseCount", len(rendered_cases))
    payload.setdefault("suiteId", context.suite.suite_id)
    payload.setdefault("subjectId", context.subject.subject_id)
    return payload


def build_case_progress(rendered_cases: list[BenchmarkCase], status: dict[str, Any],
                        results: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    results = results or {}
    case_runs_by_case_id = {
        str(case_run.get("caseId") or ""): case_run
        for case_run in results.get("caseRuns") or []
        if isinstance(case_run, dict) and case_run.get("caseId")
    }
    failed_by_case_id = {
        str(failed.get("caseId") or ""): failed
        for failed in results.get("failedCases") or []
        if isinstance(failed, dict) and failed.get("caseId")
    }
    status_failed_ids = set(str(value) for value in status.get("failedCaseIds") or [])
    failure_messages = status.get("failureMessages") or {}
    run_status = str(status.get("status") or "").upper()
    terminal_run = run_status in TERMINAL_STATUSES

    progress_rows: list[dict[str, Any]] = []
    for index, case in enumerate(rendered_cases, start=1):
        case_run = case_runs_by_case_id.get(case.case_id)
        failed = failed_by_case_id.get(case.case_id)
        row = {
            "index": index,
            "caseId": case.case_id,
            "caseName": case.name,
            "caseType": str(case.attributes_json.get("caseType") or ""),
            "caseRunId": None,
            "status": "WAITING",
            "stage": _waiting_stage(run_status),
            "progress": _waiting_progress(run_status),
            "message": "waiting for service to submit this case",
        }
        if case_run:
            case_status = str(case_run.get("status") or "RUNNING").upper()
            row.update({
                "caseRunId": case_run.get("caseRunId"),
                "status": case_status,
                "stage": _case_stage(case_status, run_status, case_run),
                "progress": _case_progress_value(case_status, run_status),
                "message": _case_error(case_run) or _case_message(case_run, run_status),
            })
        elif failed or case.case_id in status_failed_ids:
            message = None
            if failed:
                message = failed.get("message")
            message = message or failure_messages.get(case.case_id)
            row.update({
                "status": "FAILED",
                "stage": "submit_failed",
                "progress": 1.0,
                "message": message or "case failed before durable caseRun was available",
            })
        elif terminal_run:
            row.update({
                "status": "INCOMPLETE",
                "stage": "missing_result",
                "progress": 1.0,
                "message": "selected case is missing from service results",
            })
        progress_rows.append(row)
    return progress_rows


def normalize_service_results(context: RunContext, rendered_cases: list[BenchmarkCase],
                              latest_status: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    case_by_id = {case.case_id: case for case in rendered_cases}
    case_results: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    metric_results = results.get("metricResultsByCaseRunId") or {}
    raw_trace = results.get("rawTraceByCaseRunId") or {}
    events = results.get("eventsByCaseRunId") or {}
    trace_summaries = results.get("traceSummariesByCaseRunId") or {}

    for case_run in results.get("caseRuns") or []:
        case_id = str(case_run.get("caseId") or "")
        case_run_id = str(case_run.get("caseRunId") or "")
        case_metric_results = metric_results.get(case_run_id) or []
        seen_case_ids.add(case_id)
        case_results.append({
            "caseId": case_id,
            "caseName": case_by_id.get(case_id).name if case_id in case_by_id else case_id,
            "caseRunId": case_run_id,
            "status": case_run.get("status") or "UNKNOWN",
            "answerOutcome": case_run.get("answerOutcome") or "UNKNOWN",
            "scoreEligibility": case_run.get("scoreEligibility"),
            "scoreEligibilityReason": case_run.get("scoreEligibilityReason"),
            "error": _case_error(case_run),
            "inputJson": case_run.get("inputJson"),
            "expectedJson": case_run.get("expectedJson"),
            "finalOutputJson": case_run.get("finalOutputJson"),
            "durationMs": _duration_ms(case_metric_results),
            "tokenUsage": _token_usage(case_metric_results),
            "metricResults": case_metric_results,
            "events": events.get(case_run_id) or [],
            "rawTrace": raw_trace.get(case_run_id) or [],
            "traceSummaries": trace_summaries.get(case_run_id) or [],
            "attributesJson": case_run.get("attributesJson"),
        })

    for failed in results.get("failedCases") or []:
        case_id = str(failed.get("caseId") or "")
        seen_case_ids.add(case_id)
        case_results.append({
            "caseId": case_id,
            "caseName": case_by_id.get(case_id).name if case_id in case_by_id else case_id,
            "status": "FAILED",
            "answerOutcome": "NO_OUTPUT",
            "scoreEligibility": "UNSCORABLE",
            "scoreEligibilityReason": failed.get("message") or "failed before durable caseRun was available",
            "error": failed.get("message"),
            "metricResults": [],
        })

    for case in rendered_cases:
        if case.case_id not in seen_case_ids:
            case_results.append({
                "caseId": case.case_id,
                "caseName": case.name,
                "status": "INCOMPLETE",
                "answerOutcome": "NO_OUTPUT",
                "scoreEligibility": "UNSCORABLE",
                "scoreEligibilityReason": "selected case missing from service results",
                "error": "selected case missing from service results",
                "metricResults": [],
            })

    result_completeness = dict(results.get("resultCompleteness") or {})
    if str(latest_status.get("status") or "") == "UNKNOWN_OR_EXPIRED":
        result_completeness.setdefault("hasVolatileRunState", False)

    return {
        "runId": context.run_id,
        "clientRunId": context.run_id,
        "evalRunId": results.get("evalRunId") or latest_status.get("evalRunId"),
        "suite": context.suite.suite_payload(),
        "subject": context.subject.to_payload(),
        "status": results.get("status") or latest_status.get("status") or "UNKNOWN",
        "latestStatus": latest_status,
        "resultCompleteness": result_completeness,
        "caseResults": case_results,
        "failedCases": results.get("failedCases") or [],
    }


def _case_progress_summary(case_progress: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "waiting": 0,
        "running": 0,
        "success": 0,
        "failed": 0,
        "incomplete": 0,
    }
    for row in case_progress:
        status = str(row.get("status") or "").upper()
        if status in {"SUCCESS", "COMPLETED"}:
            summary["success"] += 1
        elif status in {"FAILED", "TIMEOUT", "ERROR"}:
            summary["failed"] += 1
        elif status == "INCOMPLETE":
            summary["incomplete"] += 1
        elif status in {"WAITING", "PENDING", "QUEUED"}:
            summary["waiting"] += 1
        else:
            summary["running"] += 1
    return summary


def _waiting_stage(run_status: str) -> str:
    if run_status in {"LOCAL_PREPARING", "SUBMITTED", "RUNNING"}:
        return "queued"
    if run_status == "SUBMITTING_CASES":
        return "submit_pending"
    if run_status == "COLLECTING_TRACE":
        return "trace_pending"
    if run_status == "COMPUTING_METRICS":
        return "metric_pending"
    return "waiting"


def _waiting_progress(run_status: str) -> float:
    if run_status in {"LOCAL_PREPARING", "SUBMITTED"}:
        return 0.0
    if run_status == "RUNNING":
        return 0.1
    if run_status == "SUBMITTING_CASES":
        return 0.2
    if run_status == "COLLECTING_TRACE":
        return 0.3
    if run_status == "COMPUTING_METRICS":
        return 0.7
    return 0.0


def _case_stage(case_status: str, run_status: str, case_run: dict[str, Any]) -> str:
    if case_status in {"SUCCESS", "COMPLETED"}:
        if run_status == "COMPUTING_METRICS":
            return "metrics"
        return "completed"
    if case_status in {"FAILED", "TIMEOUT", "ERROR"}:
        return "failed"
    if case_run.get("finalOutputJson"):
        return "answer_ready"
    if run_status == "COLLECTING_TRACE":
        return "trace"
    if run_status == "COMPUTING_METRICS":
        return "metrics"
    return "running"


def _case_progress_value(case_status: str, run_status: str) -> float:
    if case_status in {"SUCCESS", "COMPLETED", "FAILED", "TIMEOUT", "ERROR"}:
        return 1.0
    if run_status == "COMPUTING_METRICS":
        return 0.85
    if run_status == "COLLECTING_TRACE":
        return 0.65
    return 0.45


def _case_message(case_run: dict[str, Any], run_status: str) -> str:
    if case_run.get("finalOutputJson"):
        return "answer is available; waiting for trace or metrics"
    if run_status == "COMPUTING_METRICS":
        return "metric computation is in progress"
    if run_status == "COLLECTING_TRACE":
        return "trace collection is in progress"
    return "caseRun is available"


def _duration_ms(metric_results: list[dict[str, Any]]) -> float | None:
    duration_metric = _metric(metric_results, "openclaw.duration")
    value = duration_metric.get("valueJson") if duration_metric else None
    if not isinstance(value, dict):
        return None
    for key in ("rootClawDurationMs", "totalDurationMs", "maxDurationMs", "averageDurationMs"):
        number = _number(value.get(key))
        if number is not None:
            return number
    return None


def _token_usage(metric_results: list[dict[str, Any]]) -> float | None:
    token_metric = _metric(metric_results, "openclaw.token_usage")
    value = token_metric.get("valueJson") if token_metric else None
    if not isinstance(value, dict):
        return None
    for key in ("totalTokens", "tokenUsage", "tokens"):
        number = _number(value.get(key))
        if number is not None:
            return number
    return None


def _metric(metric_results: list[dict[str, Any]], metric_code: str) -> dict[str, Any] | None:
    for metric in metric_results:
        if isinstance(metric, dict) and metric.get("metricCode") == metric_code:
            return metric
    return None


def _case_error(case_run: dict[str, Any]) -> str | None:
    for key in ("error", "errorMessage", "message"):
        value = case_run.get(key)
        if value:
            return str(value)
    final_output = case_run.get("finalOutputJson")
    if isinstance(final_output, dict):
        for key in ("errorMessage", "error", "message"):
            value = final_output.get(key)
            if value:
                return str(value)
    return None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _local_status(context: RunContext, status: str, message: str) -> dict[str, Any]:
    return {
        "clientRunId": context.run_id,
        "suiteId": context.suite.suite_id,
        "subjectId": context.subject.subject_id,
        "status": status,
        "message": message,
        "observedAt": _now(),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
