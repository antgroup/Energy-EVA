"""AgentX Bench scoring and leaderboard row generation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import BenchmarkCase, SUCCESS_CASE_STATUSES

FAILED_CASE_STATUSES = {"FAILED", "TIMEOUT", "ERROR", "VALIDATION_FAILED"}
INCOMPLETE_CASE_STATUSES = {"INCOMPLETE", "UNKNOWN", "UNKNOWN_OR_EXPIRED"}
SCORABLE = "SCORABLE"
UNSCORABLE = "UNSCORABLE"

FORMULA = (
    "Each metric is normalized to 0..1. Case score is the weighted average of configured "
    "dimension scores for the case type. Suite score is the average across all selected "
    "cases, so failed, timed-out, incomplete, and metric-missing cases remain in the denominator."
)


def score_run(scoring: dict[str, Any], cases: list[BenchmarkCase], normalized_results: dict[str, Any]) -> dict[str, Any]:
    case_by_id = {case.case_id: case for case in cases}
    result_by_case_id = {str(result.get("caseId")): result for result in normalized_results.get("caseResults") or []}
    subject = normalized_results.get("subject") or {}
    case_scores: list[dict[str, Any]] = []

    for case in cases:
        result = result_by_case_id.get(case.case_id)
        case_scores.append(_score_case(scoring, case, result, subject))

    selected_count = len(cases)
    total_score = sum(row["caseScore"] for row in case_scores) / selected_count if selected_count else 0.0
    dimension_scores = _aggregate_dimensions(scoring, case_scores)
    metric_summary = _metric_summary(scoring, normalized_results, case_scores)
    evidence_summary = _evidence_summary(normalized_results)
    success_count = sum(1 for row in case_scores if row["status"] in SUCCESS_CASE_STATUSES)
    scorable_non_success_count = sum(1 for row in case_scores if row.get("scorableNonSuccess"))
    scored_count = sum(1 for row in case_scores if row.get("scoreEligibility") == SCORABLE)
    runtime_failed_count = sum(1 for row in case_scores if row["status"] in FAILED_CASE_STATUSES)
    unscorable_failed_count = sum(
        1 for row in case_scores
        if row["status"] in FAILED_CASE_STATUSES and row.get("scoreEligibility") != SCORABLE
    )
    incomplete_count = sum(1 for row in case_scores if row["status"] in INCOMPLETE_CASE_STATUSES)
    avg_duration = _avg([row.get("durationMs") for row in case_scores])
    token_usage = sum(float(row.get("tokenUsage") or 0.0) for row in case_scores)
    status = str(normalized_results.get("status") or "UNKNOWN")

    summary = {
        "runId": normalized_results.get("runId"),
        "clientRunId": normalized_results.get("clientRunId"),
        "evalRunId": normalized_results.get("evalRunId"),
        "suiteId": (normalized_results.get("suite") or {}).get("suiteId"),
        "suiteVersion": (normalized_results.get("suite") or {}).get("suiteVersion"),
        "subjectId": subject.get("subjectId"),
        "subjectName": subject.get("subjectName"),
        "subjectType": subject.get("subjectType"),
        "agentMode": subject.get("agentMode"),
        "status": status,
        "totalScore": round(total_score, 4),
        "dimensionScores": dimension_scores,
        "outcomeScore": dimension_scores.get("outcome", 0.0),
        "routingScore": dimension_scores.get("routing", 0.0),
        "executionScore": dimension_scores.get("execution", 0.0),
        "orchestrationScore": dimension_scores.get("orchestration", 0.0),
        "efficiencyScore": dimension_scores.get("efficiency", 0.0),
        "successRate": round(success_count / selected_count, 4) if selected_count else 0.0,
        "avgDurationMs": round(avg_duration, 2) if avg_duration is not None else None,
        "tokenUsage": round(token_usage, 2),
        "selectedCases": selected_count,
        "completedCases": success_count,
        "failedCases": runtime_failed_count,
        "runtimeSuccessCases": success_count,
        "runtimeFailedCases": runtime_failed_count,
        "scoredCases": scored_count,
        "scorableNonSuccessCases": scorable_non_success_count,
        "unscorableFailedCases": unscorable_failed_count,
        "incompleteCases": incomplete_count,
        "caseScores": case_scores,
        "metricSummary": metric_summary,
        "scoreConfidence": metric_summary.get("scoreConfidence"),
        "evidenceSummary": evidence_summary,
        "formula": FORMULA,
        "resultCompleteness": normalized_results.get("resultCompleteness") or {},
    }
    summary.update(_result_indicator_values(metric_summary))
    summary["leaderboardEligible"] = _leaderboard_eligible(summary)
    summary["leaderboardRow"] = leaderboard_row_from_summary(summary)
    return summary


def leaderboard_row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject": summary.get("subjectName") or summary.get("subjectId"),
        "subjectId": summary.get("subjectId"),
        "subjectType": summary.get("subjectType"),
        "agentMode": summary.get("agentMode"),
        "runId": summary.get("runId"),
        "evalRunId": summary.get("evalRunId"),
        "status": summary.get("status"),
        "totalScore": summary.get("totalScore"),
        "outcomeScore": summary.get("outcomeScore"),
        "routingScore": summary.get("routingScore"),
        "executionScore": summary.get("executionScore"),
        "orchestrationScore": summary.get("orchestrationScore"),
        "efficiencyScore": summary.get("efficiencyScore"),
        "successRate": summary.get("successRate"),
        "avgDurationMs": summary.get("avgDurationMs"),
        "tokenUsage": summary.get("tokenUsage"),
        "failedCases": summary.get("failedCases"),
        "runtimeSuccessCases": summary.get("runtimeSuccessCases"),
        "runtimeFailedCases": summary.get("runtimeFailedCases"),
        "scoredCases": summary.get("scoredCases"),
        "scorableNonSuccessCases": summary.get("scorableNonSuccessCases"),
        "unscorableFailedCases": summary.get("unscorableFailedCases"),
        "selectedCases": summary.get("selectedCases"),
        "completedCases": summary.get("completedCases"),
        "incompleteCases": summary.get("incompleteCases"),
        "qualityExpected": summary.get("qualityExpected"),
        "qualityScored": summary.get("qualityScored"),
        "qualityUnavailable": summary.get("qualityUnavailable"),
        "qualityMissing": summary.get("qualityMissing"),
        "qualityNotApplicable": summary.get("qualityNotApplicable"),
        "derivedExpected": summary.get("derivedExpected"),
        "derivedAvailable": summary.get("derivedAvailable"),
        "derivedMissing": summary.get("derivedMissing"),
        "scoreConfidence": summary.get("scoreConfidence"),
        "leaderboardEligible": summary.get("leaderboardEligible"),
    }


def case_result_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_score in summary.get("caseScores") or []:
        dimension_scores = case_score.get("dimensionScores") or {}
        rows.append({
            "caseId": case_score.get("caseId"),
            "caseName": case_score.get("caseName"),
            "caseType": case_score.get("caseType"),
            "status": case_score.get("status"),
            "answerOutcome": case_score.get("answerOutcome"),
            "scoreEligibility": case_score.get("scoreEligibility"),
            "scoreEligibilityReason": case_score.get("scoreEligibilityReason"),
            "scorableNonSuccess": case_score.get("scorableNonSuccess"),
            "caseScore": case_score.get("caseScore"),
            "outcomeScore": dimension_scores.get("outcome"),
            "routingScore": dimension_scores.get("routing"),
            "executionScore": dimension_scores.get("execution"),
            "orchestrationScore": dimension_scores.get("orchestration"),
            "efficiencyScore": dimension_scores.get("efficiency"),
            "durationMs": case_score.get("durationMs"),
            "tokenUsage": case_score.get("tokenUsage"),
            "reason": "; ".join(case_score.get("reasons") or []),
        })
    return rows


def _score_case(scoring: dict[str, Any], case: BenchmarkCase, result: dict[str, Any] | None,
                subject: dict[str, Any]) -> dict[str, Any]:
    case_type = str(case.attributes_json.get("caseType") or "business_answer")
    status = str((result or {}).get("status") or "INCOMPLETE").upper()
    duration_ms = _number((result or {}).get("durationMs"))
    token_usage = _token_usage(result or {})
    answer_outcome = str((result or {}).get("answerOutcome") or "UNKNOWN").upper()
    score_eligibility = str((result or {}).get("scoreEligibility") or "").upper()
    score_eligibility_reason = (result or {}).get("scoreEligibilityReason")
    reasons: list[str] = []
    metric_index = _metric_index(result or {})
    if status not in SUCCESS_CASE_STATUSES:
        score_eligibility = score_eligibility or UNSCORABLE
        if score_eligibility != SCORABLE:
            reasons.append(f"case status is {status}; scoreEligibility={score_eligibility}; case score forced to 0")
        elif not _has_useful_final_output(result or {}):
            reasons.append(f"case status is {status}; scoreEligibility=SCORABLE but final output is missing")
            score_eligibility = UNSCORABLE
        elif not _has_score_bearing_quality_metric(scoring, metric_index):
            reasons.append(f"case status is {status}; scoreEligibility=SCORABLE but score-bearing metrics are missing")
            score_eligibility = UNSCORABLE
        else:
            reasons.append(f"case status is {status}; scoreEligibility=SCORABLE; metric-derived score used")
        if score_eligibility != SCORABLE:
            error_message = _case_error(result or {})
            if error_message:
                reasons.append(error_message)
            return {
                "caseId": case.case_id,
                "caseName": case.name,
                "caseType": case_type,
                "status": status,
                "answerOutcome": answer_outcome,
                "scoreEligibility": score_eligibility,
                "scoreEligibilityReason": score_eligibility_reason,
                "scorableNonSuccess": False,
                "caseScore": 0.0,
                "dimensionScores": {},
                "dimensionDetails": {},
                "durationMs": duration_ms,
                "tokenUsage": token_usage,
                "reasons": reasons,
            }
        error_message = _case_error(result or {})
        if error_message:
            reasons.append(error_message)
    elif not score_eligibility:
        score_eligibility = SCORABLE

    dimensions = scoring.get("dimensions") or {}
    case_weights = (scoring.get("caseTypeWeights") or {}).get(case_type) or {}
    dimension_scores: dict[str, float] = {}
    dimension_details: dict[str, Any] = {}
    weighted_total = 0.0
    total_weight = 0.0

    for dimension_key, dimension_config in dimensions.items():
        dimension_weight = float(case_weights.get(dimension_key, dimension_config.get("weight", 0.0)) or 0.0)
        if dimension_weight <= 0:
            continue
        score_value, applicable, detail = _dimension_score(
            dimension_key, dimension_config, metric_index, case, result or {}, subject
        )
        dimension_details[dimension_key] = detail
        if not applicable:
            reasons.append(f"{dimension_key} not applicable")
            continue
        dimension_scores[dimension_key] = round(score_value * 100, 4)
        weighted_total += score_value * dimension_weight
        total_weight += dimension_weight

    case_score = (weighted_total / total_weight * 100) if total_weight else 0.0
    return {
        "caseId": case.case_id,
        "caseName": case.name,
        "caseType": case_type,
        "status": status,
        "answerOutcome": answer_outcome,
        "scoreEligibility": score_eligibility,
        "scoreEligibilityReason": score_eligibility_reason,
        "scorableNonSuccess": status not in SUCCESS_CASE_STATUSES and score_eligibility == SCORABLE,
        "caseScore": round(case_score, 4),
        "dimensionScores": dimension_scores,
        "dimensionDetails": dimension_details,
        "durationMs": duration_ms,
        "tokenUsage": token_usage,
        "reasons": reasons,
    }


def _dimension_score(dimension_key: str, dimension_config: dict[str, Any], metric_index: dict[str, dict[str, Any]],
                     case: BenchmarkCase, result: dict[str, Any], subject: dict[str, Any]) -> tuple[float, bool, dict[str, Any]]:
    metric_weights = dict(dimension_config.get("metrics") or {})
    metric_weights.update(dimension_config.get("derivedMetrics") or {})
    metric_scores: dict[str, float | None] = {}
    metric_reasons: dict[str, str] = {}
    weighted_total = 0.0
    total_weight = 0.0

    for metric_code, weight in metric_weights.items():
        weight_value = float(weight or 0.0)
        if weight_value <= 0:
            continue
        score_value, applicable, reason = _metric_score(metric_code, metric_index, case, result, subject)
        metric_scores[metric_code] = None if score_value is None else round(score_value, 4)
        metric_reasons[metric_code] = reason
        if not applicable:
            continue
        weighted_total += (score_value or 0.0) * weight_value
        total_weight += weight_value

    if total_weight == 0:
        return 0.0, False, {"metricScores": metric_scores, "reasons": metric_reasons}
    return weighted_total / total_weight, True, {"metricScores": metric_scores, "reasons": metric_reasons}


def _metric_score(metric_code: str, metric_index: dict[str, dict[str, Any]], case: BenchmarkCase,
                  result: dict[str, Any], subject: dict[str, Any]) -> tuple[float | None, bool, str]:
    if metric_code in metric_index:
        metric = metric_index[metric_code]
        compute_mode = str(metric.get("computeMode") or "").upper()
        if compute_mode == "NOT_APPLICABLE":
            return None, False, "metric not applicable"
        score = _number(metric.get("score"))
        if score is None:
            return 0.0, True, "metric score missing; zero by policy"
        return max(0.0, min(1.0, score)), True, str(metric.get("reason") or "metric score")

    if metric_code == "intent.match":
        return _intent_match(case, result), True, "derived from expected intent and output intent"
    if metric_code == "slots.match":
        return _slots_match(case, result), True, "derived from expected slots and output slots"
    if metric_code == "duration.normalized":
        score = _duration_score(_number(result.get("durationMs")))
        return score, True, "derived from latencyMs"
    if metric_code == "token.normalized":
        score = _token_score(_token_usage(result))
        return score, True, "derived from token usage"
    if metric_code.startswith("multi_agent.") and str(subject.get("agentMode") or "").lower() == "single_agent":
        return None, False, "single_agent subject: orchestration metric excluded"
    return 0.0, True, "metric missing; zero by policy"


def _metric_index(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for metric in result.get("metricResults") or []:
        if isinstance(metric, dict) and metric.get("metricCode"):
            indexed[str(metric["metricCode"])] = metric
    return indexed


def _has_score_bearing_quality_metric(scoring: dict[str, Any], metric_index: dict[str, dict[str, Any]]) -> bool:
    expected = _expected_score_bearing(scoring, [])
    quality_codes = set(expected["qualityMetricCodes"])
    return any(metric_code in quality_codes for metric_code in metric_index)


def _has_useful_final_output(result: dict[str, Any]) -> bool:
    output = _final_output(result)
    if output is None:
        return False
    if isinstance(output, str):
        return bool(output.strip())
    if isinstance(output, dict):
        if _is_error_fallback_output(output):
            return False
        for key in ("answer", "finalAnswer", "finalOutput", "output", "content", "text", "message", "summary"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, (dict, list)) and value:
                return True
        return bool(output.get("questions") or output.get("taskResults"))
    if isinstance(output, list):
        return bool(output)
    return True


def _is_error_fallback_output(output: dict[str, Any]) -> bool:
    error_message = str(output.get("errorMessage") or "").strip()
    if not error_message:
        return False
    answer = str(output.get("answer") or "").strip()
    only_fallback_answer = (
        not answer
        or answer == error_message
        or answer in {"编排入口执行失败", "编排入口未返回可展示结果"}
    )
    return only_fallback_answer and not output.get("summary") and not output.get("taskResults") and not output.get("questions")


def _is_metric_covered_result(case_result: dict[str, Any]) -> bool:
    status = str(case_result.get("status") or "").upper()
    if status in SUCCESS_CASE_STATUSES:
        return True
    return str(case_result.get("scoreEligibility") or "").upper() == SCORABLE


def _is_metric_covered_case_score(case_score: dict[str, Any]) -> bool:
    if case_score.get("status") in SUCCESS_CASE_STATUSES:
        return True
    return case_score.get("scoreEligibility") == SCORABLE


def _metric_summary(scoring: dict[str, Any], normalized_results: dict[str, Any],
                    case_scores: list[dict[str, Any]]) -> dict[str, Any]:
    metric_results = [
        metric
        for case_result in normalized_results.get("caseResults") or []
        for metric in case_result.get("metricResults") or []
        if isinstance(metric, dict)
    ]
    outcome_counts: defaultdict[str, int] = defaultdict(int)
    by_metric: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for metric in metric_results:
        metric_code = str(metric.get("metricCode") or "UNKNOWN")
        outcome = _metric_outcome(metric)
        outcome_counts[outcome] += 1
        by_metric[metric_code][outcome] += 1

    expected = _expected_score_bearing(scoring, case_scores)
    observed_by_case = {
        str(case_result.get("caseId")): _metric_index(case_result)
        for case_result in normalized_results.get("caseResults") or []
        if _is_metric_covered_result(case_result)
    }
    scored_quality = 0
    unavailable_quality = 0
    not_applicable_quality = 0
    missing_quality = 0
    for case_score in case_scores:
        if not _is_metric_covered_case_score(case_score):
            continue
        metric_index = observed_by_case.get(str(case_score.get("caseId"))) or {}
        for metric_code in expected["qualityMetricCodes"]:
            metric = metric_index.get(metric_code)
            if metric is None:
                missing_quality += 1
                continue
            outcome = _metric_outcome(metric)
            if outcome == "scored":
                scored_quality += 1
            elif outcome == "not_applicable":
                not_applicable_quality += 1
            elif outcome == "unavailable":
                unavailable_quality += 1
            else:
                missing_quality += 1

    derived_available = 0
    derived_missing = 0
    for case_score in case_scores:
        if not _is_metric_covered_case_score(case_score):
            continue
        if _number(case_score.get("durationMs")) is None:
            derived_missing += 1
        else:
            derived_available += 1
        if _number(case_score.get("tokenUsage")) is None:
            derived_missing += 1
        else:
            derived_available += 1

    quality_expected = scored_quality + unavailable_quality + not_applicable_quality + missing_quality
    score_confidence = _score_confidence(
        quality_expected=quality_expected,
        scored_quality=scored_quality,
        unavailable_quality=unavailable_quality,
        missing_quality=missing_quality,
    )
    return {
        "totalMetricResults": len(metric_results),
        "outcomes": dict(sorted(outcome_counts.items())),
        "byMetric": {metric_code: dict(sorted(counts.items())) for metric_code, counts in sorted(by_metric.items())},
        "qualityExpected": quality_expected,
        "qualityScored": scored_quality,
        "qualityUnavailable": unavailable_quality,
        "qualityNotApplicable": not_applicable_quality,
        "qualityMissing": missing_quality,
        "derivedExpected": derived_available + derived_missing,
        "derivedAvailable": derived_available,
        "derivedMissing": derived_missing,
        "scoreConfidence": score_confidence,
        "scoreBearingMetricCodes": expected["scoreBearingMetricCodes"],
        "qualityMetricCodes": expected["qualityMetricCodes"],
    }


def _result_indicator_values(metric_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "qualityExpected": metric_summary.get("qualityExpected"),
        "qualityScored": metric_summary.get("qualityScored"),
        "qualityUnavailable": metric_summary.get("qualityUnavailable"),
        "qualityMissing": metric_summary.get("qualityMissing"),
        "qualityNotApplicable": metric_summary.get("qualityNotApplicable"),
        "derivedExpected": metric_summary.get("derivedExpected"),
        "derivedAvailable": metric_summary.get("derivedAvailable"),
        "derivedMissing": metric_summary.get("derivedMissing"),
    }


def _expected_score_bearing(scoring: dict[str, Any], case_scores: list[dict[str, Any]]) -> dict[str, list[str]]:
    normalization = scoring.get("normalizationContract") or {}
    score_bearing = [str(item) for item in normalization.get("scoreBearingMetricCodes") or []]
    if not score_bearing:
        for dimension in (scoring.get("dimensions") or {}).values():
            score_bearing.extend(str(code) for code in (dimension.get("metrics") or {}).keys())
            score_bearing.extend(str(code) for code in (dimension.get("derivedMetrics") or {}).keys())
    derived_codes = set(((normalization.get("derivedScoreInputs") or {}).keys()))
    quality_codes = [code for code in _unique(score_bearing) if code not in derived_codes]
    return {
        "scoreBearingMetricCodes": _unique(score_bearing),
        "qualityMetricCodes": quality_codes,
    }


def _metric_outcome(metric: dict[str, Any]) -> str:
    value_json = metric.get("valueJson") if isinstance(metric.get("valueJson"), dict) else {}
    compute_mode = str(metric.get("computeMode") or value_json.get("computeMode") or "").upper()
    result_type = str(value_json.get("resultType") or "").upper()
    score_meaning = str(value_json.get("scoreMeaning") or "").upper()
    failure_type = str(value_json.get("failureType") or metric.get("errorCode") or "").upper()
    reason = str(metric.get("reason") or value_json.get("reason") or "").upper()
    if compute_mode == "NOT_APPLICABLE" or result_type == "NOT_APPLICABLE":
        return "not_applicable"
    if _number(metric.get("score")) is not None:
        return "scored"
    if "UNAVAILABLE" in {compute_mode, result_type, score_meaning} or "UNAVAILABLE" in failure_type:
        return "unavailable"
    if "FAILED" in failure_type or "ERROR" in failure_type or "FAILED" in reason or "ERROR" in reason:
        return "failed"
    if isinstance(metric.get("valueJson"), dict):
        return "statistic"
    return "skipped"


def _score_confidence(quality_expected: int, scored_quality: int,
                      unavailable_quality: int, missing_quality: int) -> str:
    if quality_expected <= 0:
        return "diagnostic"
    if scored_quality == 0 and (unavailable_quality or missing_quality):
        return "diagnostic"
    if unavailable_quality or missing_quality:
        return "limited"
    return "full"


def _leaderboard_eligible(summary: dict[str, Any]) -> bool:
    status_value = str(summary.get("status") or "").upper()
    if status_value not in {"COMPLETED", "SUCCESS"}:
        return False
    if int(summary.get("unscorableFailedCases") or 0) or int(summary.get("incompleteCases") or 0):
        return False
    if summary.get("scoreConfidence") == "diagnostic":
        return False
    return summary.get("totalScore") is not None


def _case_error(result: dict[str, Any]) -> str | None:
    error = result.get("error") or result.get("errorMessage")
    if error:
        return str(error)
    final_output = result.get("finalOutputJson")
    if isinstance(final_output, dict):
        error = final_output.get("errorMessage") or final_output.get("error")
        if error:
            return str(error)
    return None


def _evidence_summary(normalized_results: dict[str, Any]) -> dict[str, Any]:
    case_results = normalized_results.get("caseResults") or []
    event_count = 0
    raw_trace_count = 0
    trace_success = 0
    trace_failed = 0
    final_answer_count = 0
    for case_result in case_results:
        event_count += len(case_result.get("events") or [])
        raw_trace_count += len(case_result.get("rawTrace") or [])
        for summary in case_result.get("traceSummaries") or []:
            status = str(summary.get("status") or "").upper()
            if status == "SUCCESS":
                trace_success += 1
            elif status == "FAILED":
                trace_failed += 1
        final_output = case_result.get("finalOutputJson")
        if isinstance(final_output, dict) and (final_output.get("answer") or final_output.get("summary")):
            final_answer_count += 1
        elif case_result.get("output"):
            final_answer_count += 1
    return {
        "caseResultCount": len(case_results),
        "finalAnswerCount": final_answer_count,
        "eventCount": event_count,
        "rawTraceCount": raw_trace_count,
        "traceSuccessCount": trace_success,
        "traceFailedCount": trace_failed,
    }


def _intent_match(case: BenchmarkCase, result: dict[str, Any]) -> float:
    expected = case.expected_json.get("intent")
    output = _final_output(result)
    observed = output.get("intent") if isinstance(output, dict) else None
    if expected is None or observed is None:
        return 0.0
    return 1.0 if str(expected) == str(observed) else 0.0


def _slots_match(case: BenchmarkCase, result: dict[str, Any]) -> float:
    expected_slots = case.expected_json.get("slots")
    output = _final_output(result)
    observed_slots = output.get("slots") if isinstance(output, dict) else None
    if not isinstance(expected_slots, dict) or not isinstance(observed_slots, dict):
        return 0.0
    comparable = {key: value for key, value in expected_slots.items() if value is not None}
    if not comparable:
        return 1.0
    matched = sum(1 for key, value in comparable.items() if str(observed_slots.get(key)) == str(value))
    return matched / len(comparable)


def _duration_score(duration_ms: float | None) -> float:
    if duration_ms is None:
        return 0.0
    if duration_ms <= 60_000:
        return 1.0
    if duration_ms >= 600_000:
        return 0.2
    return max(0.2, 1.0 - ((duration_ms - 60_000) / 540_000) * 0.8)


def _token_score(token_usage: float | None) -> float:
    if token_usage is None:
        return 0.0
    if token_usage <= 4_000:
        return 1.0
    if token_usage >= 40_000:
        return 0.2
    return max(0.2, 1.0 - ((token_usage - 4_000) / 36_000) * 0.8)


def _final_output(result: dict[str, Any]) -> Any:
    return result.get("finalOutputJson") or result.get("output") or {}


def _token_usage(result: dict[str, Any]) -> float | None:
    direct = _number(result.get("tokenUsage"))
    if direct is not None:
        return direct
    total = 0.0
    found = False
    for metric in result.get("metricResults") or []:
        value = metric.get("valueJson") if isinstance(metric, dict) else None
        if isinstance(value, dict):
            for key in ("tokenUsage", "totalTokens", "tokens"):
                number = _number(value.get(key))
                if number is not None:
                    total += number
                    found = True
    return total if found else None


def _aggregate_dimensions(scoring: dict[str, Any], case_scores: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    dimensions = (scoring.get("dimensions") or {}).keys()
    for case_score in case_scores:
        dimension_scores = case_score.get("dimensionScores") or {}
        for dimension in dimensions:
            totals[dimension] += float(dimension_scores.get(dimension) or 0.0)
            counts[dimension] += 1
    return {dimension: round(totals[dimension] / counts[dimension], 4) if counts[dimension] else 0.0
            for dimension in dimensions}


def _avg(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
