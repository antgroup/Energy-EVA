#!/usr/bin/env python3
"""Run AgentX Bench evaluations from the command line."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx_bench.evaluation_methods.energent_swarm_service import EnergentSwarmServiceRunner
from agentx_bench.evaluation_utils.artifacts import ArtifactWriter
from agentx_bench.evaluation_utils.case_loader import (
    agentx_bench_root,
    load_suite,
    metric_codes_from_scoring,
    resolve_dataset,
    suite_from_console_detail,
)
from agentx_bench.evaluation_utils.client import AgentXClient, AgentXServiceError, parse_headers
from agentx_bench.evaluation_utils.models import RunContext, Subject
from agentx_bench.evaluation_utils.progress import (
    TerminalProgressReporter,
    build_dashboard,
    build_run_diagnostics,
)
from agentx_bench.evaluation_utils.rendering import render_cases
from agentx_bench.evaluation_utils.scoring import case_result_rows, score_run


METHOD_ENERGENT_SWARM = "energent_swarm"
METHOD_ENERGENT_SWARM_SERVICE = "energent_swarm_service"

METHOD_RUNNERS = {
    METHOD_ENERGENT_SWARM: METHOD_ENERGENT_SWARM_SERVICE,
    METHOD_ENERGENT_SWARM_SERVICE: METHOD_ENERGENT_SWARM_SERVICE,
}

DEFAULT_SUBJECTS = {
    METHOD_ENERGENT_SWARM: {
        "subject_id": "energent-swarm",
        "subject_name": "Energent Swarm",
        "subject_type": "energent_swarm",
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_dir or args.output).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    runner_method = resolve_runner_method(args.method)
    subject = resolve_subject(args, parser)
    client = None
    if runner_method == METHOD_ENERGENT_SWARM_SERVICE and args.service_url:
        client = AgentXClient(args.service_url, timeout=args.request_timeout, headers=parse_headers(args.header))
    try:
        suite = _load_suite(args, client)
        selected_cases = _select_cases(suite.cases, args.case_id, args.case_limit)
    except Exception as exc:
        print(f"AgentX Bench failed before run start: {exc}", file=sys.stderr)
        return 1
    run_id = args.run_id or _default_run_id(subject.subject_id)
    metric_codes = _metric_codes(args.metric_codes, suite.scoring)
    context = RunContext(
        run_id=run_id,
        suite=suite,
        subject=subject,
        metric_codes=metric_codes,
        output_root=output_root,
    )
    writer = ArtifactWriter(context)
    reporter = TerminalProgressReporter(enabled=not args.no_progress, mode=args.progress_mode, color=args.color)
    rendered_cases = []

    try:
        rendered_cases = render_cases(selected_cases, subject)
        reporter.start(
            context,
            len(rendered_cases),
            args.method,
            writer.run_dir,
            service_url=args.service_url,
            timeout_seconds=args.timeout,
            poll_interval_seconds=args.poll_interval,
            cases=rendered_cases,
        )
        if not reporter.confirm_start():
            print("AgentX Bench canceled before submission.", file=sys.stderr)
            return 130
        if runner_method == METHOD_ENERGENT_SWARM_SERVICE:
            if not args.service_url:
                raise ValueError("--service-url is required for energent_swarm")
            if client is None:
                client = AgentXClient(args.service_url, timeout=args.request_timeout, headers=parse_headers(args.header))
            runner = EnergentSwarmServiceRunner(
                client,
                timeout_seconds=args.timeout,
                poll_interval_seconds=args.poll_interval,
                include_events=args.include_events,
                include_metric_results=not args.exclude_metric_results,
                include_raw_trace=args.include_raw_trace,
                progress_callback=reporter.status,
            )
        else:
            raise ValueError(f"unsupported method: {args.method}")

        normalized_results = runner.run(context, rendered_cases, writer)
        reporter.results_ready(normalized_results)
        summary = score_run(suite.scoring, rendered_cases, normalized_results)
        latest_status = normalized_results.get("latestStatus") or {"status": normalized_results.get("status")}
        summary["latestStatus"] = latest_status
        summary["diagnostics"] = build_run_diagnostics(latest_status, selected_cases=len(rendered_cases))
        case_rows = case_result_rows(summary)
        writer.write_json("summary.json", summary)
        writer.write_jsonl("case_results.jsonl", case_rows)
        writer.write_csv("case_results.csv", case_rows)
        writer.write_csv("leaderboard_row.csv", [summary["leaderboardRow"]])
        dashboard = build_dashboard(summary, writer.run_dir)
        writer.write_text("dashboard.txt", dashboard)
        writer.write_text("diagnostic_report.txt", dashboard)
        if not args.no_dashboard:
            if not reporter.complete(summary, writer.run_dir, dashboard_enabled=True):
                print(writer.run_dir)
                return 130
            reporter.dashboard(dashboard)
        print(writer.run_dir)
        return 0
    except KeyboardInterrupt as exc:
        return _handle_failure(
            context=context,
            writer=writer,
            reporter=reporter,
            selected_cases=selected_cases,
            rendered_cases=rendered_cases,
            error=exc,
            status="INTERRUPTED",
            exit_code=130,
            no_dashboard=args.no_dashboard,
            message="interrupted by user",
        )
    except Exception as exc:
        return _handle_failure(
            context=context,
            writer=writer,
            reporter=reporter,
            selected_cases=selected_cases,
            rendered_cases=rendered_cases,
            error=exc,
            status="TIMEOUT" if isinstance(exc, TimeoutError) else "FAILED",
            exit_code=124 if isinstance(exc, TimeoutError) else 1,
            no_dashboard=args.no_dashboard,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentX Bench.")
    parser.add_argument("--suite", default="power_trade_standard_v1", help="Dataset suite name.")
    parser.add_argument("--scoring-suite", default="power_trade_standard_v1",
                        help="Local scoring suite used when --suite is loaded from the Evaluation Console API.")
    parser.add_argument("--dataset", help="Explicit dataset directory.")
    parser.add_argument("--method", default=METHOD_ENERGENT_SWARM,
                        choices=[METHOD_ENERGENT_SWARM, METHOD_ENERGENT_SWARM_SERVICE])
    parser.add_argument("--service-url", help="AgentX Evaluation Service base URL.")
    parser.add_argument("--subject-id")
    parser.add_argument("--subject-name")
    parser.add_argument("--subject-type")
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--agent-mode", default="multi_agent", choices=["single_agent", "multi_agent"])
    parser.add_argument("--model")
    parser.add_argument("--subject-version")
    parser.add_argument("--output", default="evaluation_results/agentx", help="Artifact output root.")
    parser.add_argument("--output-dir", default=None, help="Alias for --output.")
    parser.add_argument("--run-id", help="Override local run id.")
    parser.add_argument("--metric-codes", help="Comma separated metric codes sent to the service.")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--header", action="append", help='HTTP header, for example "Authorization: Bearer xxx".')
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--exclude-metric-results", action="store_true")
    parser.add_argument("--include-raw-trace", action="store_true")
    parser.add_argument("--case-id", action="append", help="Run only selected caseId; can repeat.")
    parser.add_argument("--case-limit", type=int, help="Run the first N cases; intended for smoke tests.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable the professional progress UI and interactive confirmations.")
    parser.add_argument("--progress-mode", default="auto", choices=["auto", "panel", "lines"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--color", default="auto", choices=["auto", "always", "never"],
                        help="Color mode for terminal rendering.")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable final terminal dashboard output and its interactive confirmation.")
    return parser


def resolve_runner_method(method: str) -> str:
    return METHOD_RUNNERS.get(method, method)


def resolve_subject(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> Subject:
    defaults = DEFAULT_SUBJECTS.get(args.method, {})
    subject_id = args.subject_id or defaults.get("subject_id")
    subject_name = args.subject_name or defaults.get("subject_name")
    subject_type = args.subject_type or defaults.get("subject_type")
    missing = []
    if not subject_id:
        missing.append("--subject-id")
    if not subject_name:
        missing.append("--subject-name")
    if not subject_type:
        missing.append("--subject-type")
    if missing:
        message = f"{', '.join(missing)} required unless --method {METHOD_ENERGENT_SWARM} is used"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return Subject(
        subject_id=subject_id,
        subject_name=subject_name,
        subject_type=subject_type,
        team_name=args.team_name,
        agent_mode=args.agent_mode,
        model=args.model,
        version=args.subject_version,
    )


def _load_suite(args: argparse.Namespace, client: AgentXClient | None):
    if args.dataset:
        return load_suite(resolve_dataset(args.suite, args.dataset))
    try:
        return load_suite(resolve_dataset(args.suite, args.dataset))
    except FileNotFoundError as local_exc:
        if client is None:
            datasets_dir = agentx_bench_root() / "datasets"
            raise FileNotFoundError(
                f"dataset suite not found locally: {args.suite}. "
                f"Local suites live under {datasets_dir}. "
                "If this is an Evaluation Console suite, start the host service and pass --service-url; "
                "otherwise use a local suite such as power_trade_standard_v1."
            ) from local_exc
        fallback_suite = load_suite(resolve_dataset(args.scoring_suite)) if args.scoring_suite else None
        try:
            detail = client.get_console_suite(args.suite)
        except AgentXServiceError as service_exc:
            raise AgentXServiceError(
                f"dataset suite '{args.suite}' was not found in local AgentX Bench datasets, "
                f"so it must be loaded from the Evaluation Console service at {client.base_url}. "
                f"The service request failed: {service_exc}. "
                "Start the host service or correct --service-url; for the built-in benchmark dataset use "
                "--suite power_trade_standard_v1."
            ) from service_exc
        return suite_from_console_detail(
            detail,
            fallback_scoring=fallback_suite.scoring if fallback_suite else {},
            fallback_suite=fallback_suite.suite if fallback_suite else {},
        )


def _metric_codes(value: str | None, scoring: dict) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return metric_codes_from_scoring(scoring)


def _select_cases(cases, case_ids: list[str] | None, case_limit: int | None):
    selected = cases
    if case_ids:
        case_id_set = set(case_ids)
        selected = [case for case in selected if case.case_id in case_id_set]
        missing = case_id_set - {case.case_id for case in selected}
        if missing:
            raise ValueError(f"unknown --case-id values: {', '.join(sorted(missing))}")
    if case_limit is not None:
        selected = selected[:case_limit]
    if not selected:
        raise ValueError("no cases selected")
    return selected


def _default_run_id(subject_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{subject_id}-{timestamp}"


def _handle_failure(context: RunContext, writer: ArtifactWriter, reporter: TerminalProgressReporter,
                    selected_cases, rendered_cases, error: BaseException, status: str, exit_code: int,
                    no_dashboard: bool, message: str | None = None) -> int:
    error_message = message or str(error)
    writer.append_error({
        "errorType": error.__class__.__name__,
        "message": error_message,
        "observedAt": datetime.now(timezone.utc).isoformat(),
    })
    latest_status = _latest_status(writer)
    failure_status = {
        **latest_status,
        "clientRunId": context.run_id,
        "suiteId": context.suite.suite_id,
        "subjectId": context.subject.subject_id,
        "status": status,
        "phase": "client_failure" if status == "FAILED" else status.lower(),
        "message": error_message,
        "observedAt": datetime.now(timezone.utc).isoformat(),
    }
    writer.append_status(failure_status)
    cases = rendered_cases or selected_cases
    normalized_results = _incomplete_results(context, cases, failure_status, status, error_message)
    writer.write_json("raw_results.json", normalized_results)
    summary = score_run(context.suite.scoring, cases, normalized_results)
    diagnostics = build_run_diagnostics(
        latest_status or failure_status,
        selected_cases=len(cases),
        error_message=error_message,
    )
    summary["status"] = status
    summary["error"] = error_message
    summary["latestStatus"] = latest_status or failure_status
    summary["diagnostics"] = diagnostics
    summary["leaderboardRow"]["status"] = status
    writer.write_json("summary.json", summary)
    case_rows = case_result_rows(summary)
    writer.write_jsonl("case_results.jsonl", case_rows)
    writer.write_csv("case_results.csv", case_rows)
    writer.write_csv("leaderboard_row.csv", [summary["leaderboardRow"]])
    dashboard = build_dashboard(summary, writer.run_dir)
    writer.write_text("dashboard.txt", dashboard)
    writer.write_text("diagnostic_report.txt", dashboard)
    if not no_dashboard:
        reporter.complete(summary, writer.run_dir, dashboard_enabled=True)
        reporter.dashboard(dashboard)
    print(f"AgentX Bench failed; artifacts: {writer.run_dir}", file=sys.stderr)
    print(error_message, file=sys.stderr)
    return exit_code


def _incomplete_results(context: RunContext, cases, latest_status: dict, status: str,
                        error_message: str) -> dict:
    failed_case_ids = set(str(value) for value in latest_status.get("failedCaseIds") or [])
    failure_messages = latest_status.get("failureMessages") or {}
    case_results = []
    for case in cases:
        case_failed = case.case_id in failed_case_ids
        case_results.append({
            "caseId": case.case_id,
            "caseName": case.name,
            "status": "FAILED" if case_failed else "INCOMPLETE",
            "answerOutcome": "NO_OUTPUT",
            "scoreEligibility": "UNSCORABLE",
            "scoreEligibilityReason": failure_messages.get(case.case_id) or error_message,
            "error": failure_messages.get(case.case_id) or error_message,
            "inputJson": case.input_json,
            "expectedJson": case.expected_json,
            "metricResults": [],
            "attributesJson": case.attributes_json,
        })
    return {
        "runId": context.run_id,
        "clientRunId": context.run_id,
        "evalRunId": latest_status.get("evalRunId"),
        "suite": context.suite.suite_payload(),
        "subject": context.subject.to_payload(),
        "status": status,
        "latestStatus": latest_status,
        "resultCompleteness": {
            "hasVolatileRunState": bool(latest_status.get("volatileStatePresent")),
            "hasSelectedCaseEcho": bool(latest_status.get("selectedCaseIds")),
            "preCaseRunFailuresComplete": bool(failed_case_ids),
            "detailFlags": {
                "hasFinalServiceResult": False,
                "hasMetricResults": False,
                "hasRawTrace": False,
            },
        },
        "caseResults": case_results,
        "failedCases": [
            {"caseId": case_id, "message": failure_messages.get(case_id) or error_message}
            for case_id in sorted(failed_case_ids)
        ],
    }


def _latest_status(writer: ArtifactWriter) -> dict:
    path = writer.path("run_status.json")
    if not path.exists():
        return {}
    try:
        import json
        with path.open("r", encoding="utf-8") as file_obj:
            value = json.load(file_obj)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
