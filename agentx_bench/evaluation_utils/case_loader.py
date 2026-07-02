"""Dataset loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import BenchmarkCase, SuiteBundle

KNOWN_LEADERBOARD_FIELDS = {
    "rank",
    "subject",
    "subjectId",
    "subjectName",
    "subjectType",
    "agentMode",
    "runId",
    "clientRunId",
    "evalRunId",
    "status",
    "totalScore",
    "successRate",
    "avgDurationMs",
    "tokenUsage",
    "failedCases",
    "selectedCases",
    "completedCases",
    "incompleteCases",
    "qualityExpected",
    "qualityScored",
    "qualityUnavailable",
    "qualityMissing",
    "qualityNotApplicable",
    "scoreConfidence",
    "derivedExpected",
    "derivedAvailable",
    "derivedMissing",
    "leaderboardEligible",
}


def agentx_bench_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset(suite: str | None = None, dataset: str | None = None) -> Path:
    if dataset:
        path = Path(dataset).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    suite_name = suite or "power_trade_standard_v1"
    candidates = [
        agentx_bench_root() / "datasets" / suite_name,
        agentx_bench_root() / "datasets" / suite_name.replace("-", "_"),
        agentx_bench_root() / "datasets" / suite_name.replace("_", "-"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"dataset suite not found: {suite_name}")


def load_suite(dataset_dir: Path) -> SuiteBundle:
    suite_path = dataset_dir / "suite.yaml"
    cases_path = dataset_dir / "cases.jsonl"
    scoring_path = dataset_dir / "scoring.yaml"
    for required_path in (suite_path, cases_path, scoring_path):
        if not required_path.exists():
            raise FileNotFoundError(f"missing dataset file: {required_path}")

    suite = _load_yaml(suite_path)
    scoring = _load_yaml(scoring_path)
    cases = _load_cases(cases_path)
    _validate_suite(dataset_dir, suite, scoring, cases)
    return SuiteBundle(dataset_dir=dataset_dir, suite=suite, cases=cases, scoring=scoring)


def suite_from_console_detail(detail: dict[str, Any], fallback_scoring: dict[str, Any] | None = None,
                              fallback_suite: dict[str, Any] | None = None) -> SuiteBundle:
    suite_key = str(detail.get("suiteKey") or detail.get("suiteId") or "").strip()
    suite_id = str(detail.get("suiteId") or suite_key).strip()
    if not suite_id:
        raise ValueError("console suite detail missing suiteId")

    raw_suite = dict(detail.get("suite") or {})
    cases = _cases_from_console_detail(detail)
    suite = _suite_from_console_detail(detail, raw_suite, cases, fallback_suite or {})

    raw_scoring = ((detail.get("scoring") or {}).get("rawScoring") or {})
    scoring = dict(raw_scoring or fallback_scoring or {})
    if not scoring.get("dimensions"):
        raise ValueError(
            f"console suite {suite_id} does not provide scoring and no fallback scoring is available"
        )
    # 服务端评测集不是本地目录，但保持 dataset_dir 可读，便于 artifact 中保留来源语义。
    return SuiteBundle(dataset_dir=Path("service") / suite_id, suite=suite, cases=cases, scoring=scoring)


def metric_codes_from_scoring(scoring: dict[str, Any]) -> list[str]:
    metric_codes: list[str] = []
    for dimension in (scoring.get("dimensions") or {}).values():
        for metric_code in (dimension.get("metrics") or {}).keys():
            if metric_code not in metric_codes:
                metric_codes.append(metric_code)
        for derived_config in (dimension.get("derivedMetrics") or {}).values():
            if isinstance(derived_config, dict):
                source_metric_code = str(derived_config.get("sourceMetricCode") or "").strip()
                if source_metric_code and source_metric_code not in metric_codes:
                    metric_codes.append(source_metric_code)
    for derived_config in ((scoring.get("normalizationContract") or {}).get("derivedScoreInputs") or {}).values():
        if not isinstance(derived_config, dict):
            continue
        source_metric_code = str(derived_config.get("sourceMetricCode") or "").strip()
        if source_metric_code and source_metric_code not in metric_codes:
            metric_codes.append(source_metric_code)
    return metric_codes


def _suite_from_console_detail(detail: dict[str, Any], raw_suite: dict[str, Any],
                               cases: list[BenchmarkCase], fallback_suite: dict[str, Any]) -> dict[str, Any]:
    suite = dict(raw_suite)
    suite_id = str(detail.get("suiteId") or detail.get("suiteKey") or suite.get("datasetId") or "").strip()
    suite["datasetId"] = suite_id
    suite["name"] = detail.get("name") or suite.get("name") or suite_id
    suite["description"] = detail.get("description") or suite.get("description") or ""
    suite["version"] = detail.get("version") or suite.get("version") or fallback_suite.get("version") or "v1"
    suite["defaultAsOfDate"] = (
        detail.get("defaultAsOfDate") or suite.get("defaultAsOfDate") or fallback_suite.get("defaultAsOfDate") or ""
    )
    suite["timezone"] = detail.get("timezone") or suite.get("timezone") or fallback_suite.get("timezone") or "Asia/Shanghai"
    suite["caseCount"] = len(cases)
    suite.setdefault("caseFiles", ["service-console"])
    case_types = suite.get("caseTypes")
    if not isinstance(case_types, dict) or not case_types:
        suite["caseTypes"] = _derive_case_types(cases)
    recommended_subjects = suite.get("recommendedSubjects")
    if not isinstance(recommended_subjects, list) or not recommended_subjects:
        suite["recommendedSubjects"] = [{
            "subjectId": "energent-swarm",
            "subjectName": "Energent Swarm",
            "subjectType": "energent_swarm",
            "agentMode": "multi_agent",
            "comparisonPath": "automatic_eval_service",
        }]
    return suite


def _cases_from_console_detail(detail: dict[str, Any]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(detail.get("cases") or [], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"console suite case #{index} must be an object")
        normalized = {
            "caseId": raw.get("caseId"),
            "name": raw.get("caseName") or raw.get("name") or raw.get("caseId"),
            "inputJson": raw.get("inputJson") or {},
            "expectedJson": raw.get("expectedJson") or {},
            "attributesJson": raw.get("attributesJson") or {},
        }
        case = BenchmarkCase.from_raw(normalized, index)
        if case.case_id in seen:
            raise ValueError(f"console suite contains duplicate caseId {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("console suite must contain at least one case")
    return cases


def _derive_case_types(cases: list[BenchmarkCase]) -> dict[str, Any]:
    case_types: dict[str, Any] = {}
    for case in cases:
        case_type = str(case.attributes_json.get("caseType") or "business_answer")
        case_types.setdefault(case_type, {"description": "从服务端评测集 Case 属性推导的类型。"})
    return case_types


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = yaml.safe_load(file_obj) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a YAML object")
    return value


def _load_cases(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"cases.jsonl line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"cases.jsonl line {line_no}: each row must be an object")
            case = BenchmarkCase.from_raw(raw, line_no)
            if case.case_id in seen:
                raise ValueError(f"cases.jsonl line {line_no}: duplicate caseId {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("cases.jsonl must contain at least one case")
    return cases


def _validate_suite(dataset_dir: Path, suite: dict[str, Any], scoring: dict[str, Any],
                    cases: list[BenchmarkCase]) -> None:
    dataset_id = suite.get("datasetId") or suite.get("suite")
    if not dataset_id:
        raise ValueError("suite.yaml must define datasetId or suite")
    if not suite.get("defaultAsOfDate"):
        raise ValueError("suite.yaml must define defaultAsOfDate")
    if suite.get("caseCount") is not None and int(suite["caseCount"]) != len(cases):
        raise ValueError(
            f"suite.yaml caseCount={suite['caseCount']} does not match cases.jsonl count={len(cases)}"
        )
    if not scoring.get("dimensions"):
        raise ValueError("scoring.yaml must define dimensions")
    case_files = suite.get("caseFiles") or ["cases.jsonl"]
    if "cases.jsonl" not in case_files:
        raise ValueError(f"{dataset_dir}: suite.yaml must include cases.jsonl in caseFiles")
    _validate_case_types(suite, scoring, cases)
    _validate_standard_metadata(suite, scoring)
    _validate_ranking_metadata(scoring)


def _validate_case_types(suite: dict[str, Any], scoring: dict[str, Any],
                         cases: list[BenchmarkCase]) -> None:
    suite_case_types = suite.get("caseTypes") or {}
    scoring_case_weights = scoring.get("caseTypeWeights") or {}
    dimensions = scoring.get("dimensions") or {}
    if not isinstance(suite_case_types, dict) or not suite_case_types:
        raise ValueError("suite.yaml must define caseTypes")
    if not isinstance(scoring_case_weights, dict) or not scoring_case_weights:
        raise ValueError("scoring.yaml must define caseTypeWeights")
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("scoring.yaml dimensions must be a non-empty object")

    suite_case_type_keys = set(suite_case_types)
    scoring_case_type_keys = set(scoring_case_weights)
    if suite_case_type_keys != scoring_case_type_keys:
        raise ValueError(
            "suite.yaml caseTypes must match scoring.yaml caseTypeWeights: "
            f"suite={sorted(suite_case_type_keys)}, scoring={sorted(scoring_case_type_keys)}"
        )

    dimension_keys = set(dimensions)
    for case_type, weights in scoring_case_weights.items():
        if not isinstance(weights, dict):
            raise ValueError(f"scoring.yaml caseTypeWeights.{case_type} must be an object")
        weight_keys = set(weights)
        unknown_dimensions = sorted(weight_keys - dimension_keys)
        missing_dimensions = sorted(dimension_keys - weight_keys)
        if unknown_dimensions:
            raise ValueError(
                f"scoring.yaml caseTypeWeights.{case_type} contains unknown dimensions: "
                f"{', '.join(unknown_dimensions)}"
            )
        if missing_dimensions:
            raise ValueError(
                f"scoring.yaml caseTypeWeights.{case_type} missing dimensions: "
                f"{', '.join(missing_dimensions)}"
            )
        total_weight = sum(_required_number(value, f"caseTypeWeights.{case_type}") for value in weights.values())
        if abs(total_weight - 100.0) > 0.0001:
            raise ValueError(f"scoring.yaml caseTypeWeights.{case_type} must sum to 100, got {total_weight:g}")

    for case in cases:
        case_type = str(case.attributes_json.get("caseType") or "")
        if not case_type:
            raise ValueError(f"cases.jsonl {case.case_id}: attributesJson.caseType is required")
        if case_type not in suite_case_types:
            raise ValueError(f"cases.jsonl {case.case_id}: unknown suite caseType {case_type}")
        if case_type not in scoring_case_weights:
            raise ValueError(f"cases.jsonl {case.case_id}: missing scoring weights for caseType {case_type}")


def _validate_standard_metadata(suite: dict[str, Any], scoring: dict[str, Any]) -> None:
    normalization = scoring.get("normalizationContract")
    if not isinstance(normalization, dict):
        raise ValueError("scoring.yaml must define normalizationContract")
    for field_name in ("requiredResultFields", "requiredMetricResultFields"):
        if not _non_empty_string_list(normalization.get(field_name)):
            raise ValueError(f"scoring.yaml normalizationContract.{field_name} must be a non-empty string list")

    auxiliary_metrics = scoring.get("auxiliaryMetrics")
    if not isinstance(auxiliary_metrics, list) or not auxiliary_metrics:
        raise ValueError("scoring.yaml must define auxiliaryMetrics")
    for index, metric in enumerate(auxiliary_metrics, start=1):
        if not isinstance(metric, dict):
            raise ValueError(f"scoring.yaml auxiliaryMetrics[{index}] must be an object")
        code = str(metric.get("code") or "").strip()
        role = str(metric.get("role") or "").strip()
        if not code:
            raise ValueError(f"scoring.yaml auxiliaryMetrics[{index}] missing code")
        if not role:
            raise ValueError(f"scoring.yaml auxiliaryMetrics[{code}] missing role")
        if "scoreBearing" not in metric or not isinstance(metric.get("scoreBearing"), bool):
            raise ValueError(f"scoring.yaml auxiliaryMetrics[{code}] must define boolean scoreBearing")
        if not str(metric.get("name") or metric.get("description") or "").strip():
            raise ValueError(f"scoring.yaml auxiliaryMetrics[{code}] needs name or description")
        if not metric["scoreBearing"] and role.lower() == "primary":
            raise ValueError(f"scoring.yaml auxiliaryMetrics[{code}] cannot be primary when scoreBearing is false")

    result_indicators = scoring.get("resultIndicators")
    if not isinstance(result_indicators, dict):
        raise ValueError("scoring.yaml must define resultIndicators")
    indicators = result_indicators.get("indicators")
    if not isinstance(indicators, list) or not indicators:
        raise ValueError("scoring.yaml resultIndicators.indicators must be a non-empty list")
    if result_indicators.get("count") is not None and int(result_indicators["count"]) != len(indicators):
        raise ValueError(
            "scoring.yaml resultIndicators.count must match indicators length: "
            f"count={result_indicators['count']}, indicators={len(indicators)}"
        )
    indicator_codes: set[str] = set()
    for index, indicator in enumerate(indicators, start=1):
        if not isinstance(indicator, dict):
            raise ValueError(f"scoring.yaml resultIndicators[{index}] must be an object")
        code = str(indicator.get("code") or "").strip()
        category = str(indicator.get("category") or "").strip()
        source = str(indicator.get("source") or "").strip()
        if not code:
            raise ValueError(f"scoring.yaml resultIndicators[{index}] missing code")
        if code in indicator_codes:
            raise ValueError(f"scoring.yaml resultIndicators contains duplicate code: {code}")
        indicator_codes.add(code)
        if code not in KNOWN_LEADERBOARD_FIELDS and not code.endswith("Score"):
            raise ValueError(f"scoring.yaml resultIndicators[{code}] references unknown summary field")
        if not category:
            raise ValueError(f"scoring.yaml resultIndicators[{code}] missing category")
        if not source:
            raise ValueError(f"scoring.yaml resultIndicators[{code}] missing source")
        if "scoreBearing" not in indicator or not isinstance(indicator.get("scoreBearing"), bool):
            raise ValueError(f"scoring.yaml resultIndicators[{code}] must define boolean scoreBearing")
        if not str(indicator.get("name") or indicator.get("description") or "").strip():
            raise ValueError(f"scoring.yaml resultIndicators[{code}] needs name or description")

    recommended_subjects = suite.get("recommendedSubjects")
    if not isinstance(recommended_subjects, list) or not recommended_subjects:
        raise ValueError("suite.yaml must define recommendedSubjects")
    required_subject_fields = ("subjectId", "subjectName", "subjectType", "agentMode")
    for index, subject in enumerate(recommended_subjects, start=1):
        if not isinstance(subject, dict):
            raise ValueError(f"suite.yaml recommendedSubjects[{index}] must be an object")
        missing = [
            field_name for field_name in required_subject_fields
            if not str(subject.get(field_name) or "").strip()
        ]
        if missing:
            raise ValueError(
                f"suite.yaml recommendedSubjects[{index}] missing fields: {', '.join(missing)}"
            )


def _validate_ranking_metadata(scoring: dict[str, Any]) -> None:
    dimensions = scoring.get("dimensions") or {}
    dimension_score_fields = {f"{dimension_key}Score" for dimension_key in dimensions.keys()}
    compatibility_columns = scoring.get("compatibilityColumns") or {}
    compatibility_fields = set(compatibility_columns.keys()) if isinstance(compatibility_columns, dict) else set()
    known_fields = KNOWN_LEADERBOARD_FIELDS | dimension_score_fields | compatibility_fields

    ranking = scoring.get("ranking") or {}
    primary = ranking.get("primary")
    if primary and _ranking_field(str(primary)) not in known_fields:
        raise ValueError(f"scoring.yaml ranking.primary references unknown summary field: {primary}")
    for tie_breaker in ranking.get("tieBreakers") or []:
        field_name = _ranking_field(str(tie_breaker))
        if field_name not in known_fields:
            raise ValueError(f"scoring.yaml ranking.tieBreakers references unknown summary field: {tie_breaker}")

    for column in scoring.get("publishColumns") or []:
        if str(column) not in known_fields:
            raise ValueError(f"scoring.yaml publishColumns references unknown summary field: {column}")

    published_ranking = scoring.get("publishedRanking")
    if not isinstance(published_ranking, dict):
        raise ValueError("scoring.yaml must define publishedRanking")
    if not _non_empty_string_list(published_ranking.get("requiredDisclosureFields")):
        raise ValueError("scoring.yaml publishedRanking.requiredDisclosureFields must be a non-empty string list")


def _ranking_field(value: str) -> str:
    for suffix in ("Asc", "Desc"):
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def _non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and any(str(item or "").strip() for item in value)


def _required_number(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scoring.yaml {field_name} weights must be numeric") from exc
