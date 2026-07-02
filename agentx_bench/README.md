# AgentX Bench: Agent Evaluation Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Scenario: Power Trading](https://img.shields.io/badge/scenario-power%20trading-1b66f2.svg)](datasets/power_trade_standard_v1)
[![Runner: Energent Swarm](https://img.shields.io/badge/runner-Energent%20Swarm-16835f.svg)](#run-energent-swarm)

## Updates

- **[2026-06-30]**: Released the frozen `power_trade_standard_v1`
  leaderboard snapshot for AgentX Bench, with Energent Swarm and multiple
  single-agent comparison rows.

## Scene Description

AgentX Bench currently supports `power_trade_standard_v1`, a power trading
agent benchmark with 43 cases covering market queries, policy rules,
forecasting, intraday strategy, retail packages, trade reviews, and
external-signal analysis.

## Overview

AgentX Bench is an agent-oriented benchmark for evaluating whether an AI agent
can complete realistic domain workflows, not only answer isolated questions.
It combines curated scenario suites, service-driven batch execution,
trace-aware metric collection, and a four-dimension scoring standard for
comparing agent products in a reproducible leaderboard format.

The currently supported scenario is customer-side electricity trading through
the `power_trade_standard_v1` suite.

The current open-source package provides a self-contained Python command-line
runner for evaluating Energent Swarm through the AgentX Evaluation Service API.
It does not require Java, SOFABoot, database access, or orchestration-module
runtime dependencies inside the benchmark package.

## Key Features

- Scenario-based agent benchmark with the first supported suite for electricity
  trading.
- 43 curated power trading business cases in the current suite.
- Workflow-readiness scoring across result quality, execution, orchestration,
  and efficiency.
- Built-in Energent Swarm service runner with live terminal progress and local
  JSON/CSV/TXT artifacts.
- Frozen leaderboard snapshot for comparing Energent Swarm with representative
  single-agent baselines.
- Portable dataset and scoring files designed to be reviewed independently from
  the host application.

## Leaderboard Snapshot

Access the static frozen ranking page at
[`frozen_results/power_trade_standard_v1/leaderboard/index.html`](frozen_results/power_trade_standard_v1/leaderboard/index.html).

| Rank | Subject | Agent Mode | totalScore | outcomeScore | executionScore | orchestrationScore | efficiencyScore | successRate | Status |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Energent Swarm | multi_agent | 68.6605 | 76.0465 | 66.9572 | 68.1395 | 16.6219 | 76.74% | COMPLETED |
| 2 | CrewAI | single_agent | 53.9729 | 46.0368 | 45.0155 | 0.0000 | 97.0928 | 100.00% | COMPLETED |
| 3 | OpenClaw | single_agent | 42.5930 | 40.2372 | 35.7829 | 0.0000 | 57.0743 | 100.00% | COMPLETED |
| 4 | Claude | single_agent | 42.1808 | 34.1714 | 27.6636 | 0.0000 | 90.8612 | 99.22% | DIAGNOSTIC |
| 5 | Hermes | single_agent | 40.1049 | 35.9200 | 23.7729 | 0.0000 | 74.6772 | 96.12% | DIAGNOSTIC |
| 6 | OpenManus | single_agent | 34.8078 | 24.1310 | 24.3446 | 0.0000 | 87.7757 | 100.00% | COMPLETED |
| 7 | Goose | single_agent | 33.1491 | 33.6964 | 6.5194 | 0.0000 | 59.8964 | 100.00% | COMPLETED |

Rows marked `DIAGNOSTIC` are retained for comparison but are not official
leaderboard-eligible snapshots. The score source files are stored under
`frozen_results/power_trade_standard_v1/<subject>/<date>/leaderboard_row.json`.

## Dataset Description

The default suite is `power_trade_standard_v1`. All relative dates are anchored
to `2026-06-23` in the benchmark metadata. Each case contains a user-visible
business query, expected answer criteria, and lightweight attributes such as
category, case type, and difficulty.

| File | Description |
| --- | --- |
| `datasets/power_trade_standard_v1/suite.yaml` | Dataset metadata, coverage, case types, and benchmark rules. |
| `datasets/power_trade_standard_v1/cases.jsonl` | 43 active benchmark cases. |
| `datasets/power_trade_standard_v1/scoring.yaml` | Official dimensions, metric weights, case-type weights, ranking fields, and publishing policy. |
| `datasets/power_trade_standard_v1/source_row_audit.jsonl` | Sanitized source-row audit records for dataset governance. |
| `datasets/power_trade_standard_v1/curation_guide.md` | Dataset curation and scoring-governance notes. |

## Scope

- Built-in runner: Energent Swarm via HTTP.
- Built-in dataset: `power_trade_standard_v1`.
- Built-in scoring: `datasets/power_trade_standard_v1/scoring.yaml`.
- Output: local JSON/CSV/TXT artifacts under `evaluation_results/agentx`.

This trimmed package does not include external-agent adapters, external-result
import flows, hosted leaderboards, database migrations, or browser automation.

## Requirements

- Python 3.10+.
- PyYAML.
- A running host application exposing the AgentX Evaluation Service API.

Install the only non-stdlib dependency when needed:

```bash
python3 -m pip install -r requirements.txt
```

## Dataset

Run commands from the `agentx_bench` directory:

```bash
cd agentx_bench
```

The default suite lives at:

```text
datasets/power_trade_standard_v1
```

Key files:

- `suite.yaml`: dataset metadata and case type definitions.
- `cases.jsonl`: active benchmark cases.
- `scoring.yaml`: scoring dimensions, metric weights, and ranking policy.
- `source_row_audit.jsonl`: sanitized source-row audit records.
- `curation_guide.md`: dataset governance notes.

Inspect the case list:

```bash
./bin/agentx-bench tasks \
  --suite power_trade_standard_v1
```

## Run Energent Swarm

Start a host application that includes the evaluation module and exposes the
AgentX API. The host must enable:

```properties
energy-ai.evaluation.agentx-api.enabled=true
```

Then run the benchmark:

```bash
./bin/agentx-bench run \
  --suite power_trade_standard_v1 \
  --method energent_swarm \
  --service-url http://127.0.0.1:8080 \
  --team-name virtual-trader-team \
  --agent-mode multi_agent \
  --output evaluation_results/agentx \
  --timeout 7200 \
  --request-timeout 120 \
  --poll-interval 30 \
  --include-events \
  --include-raw-trace
```

`--method energent_swarm` defaults the benchmark subject to:

```text
subjectId   = energent-swarm
subjectName = Energent Swarm
subjectType = energent_swarm
```

For a smoke test, add either `--case-limit 1` or one or more explicit case ids:

```bash
./bin/agentx-bench run \
  --suite power_trade_standard_v1 \
  --method energent_swarm \
  --service-url http://127.0.0.1:8080 \
  --team-name virtual-trader-team \
  --agent-mode multi_agent \
  --case-id power_trade_standard_v1_001 \
  --output evaluation_results/agentx-smoke \
  --timeout 1800 \
  --request-timeout 120 \
  --poll-interval 30
```

For CI or shell scripts, add `--no-progress --no-dashboard --color never` to
disable interactive prompts and terminal dashboards.

## Console Flow

In an interactive terminal, the CLI prints:

- the AgentX Bench banner and runtime configuration;
- target subject, team, suite, output path, and planned cases;
- live stage progress while the service runs;
- a final dashboard after scoring completes.

In non-interactive shells, the CLI starts immediately and writes the same
artifacts without waiting for keyboard input.

## Output

Each run writes to:

```text
{output}/{suiteId}/{subjectId}/{runId}/
```

Important files:

- `request.json`: submitted AgentX payload.
- `run_status.json`: latest observed service status.
- `status_history.jsonl`: status polling history.
- `raw_results.json`: normalized service results.
- `summary.json`: score summary and leaderboard row.
- `case_results.jsonl` / `case_results.csv`: per-case score rows.
- `leaderboard_row.csv`: one-row ranking export for this run.
- `dashboard.txt` / `diagnostic_report.txt`: readable run report.
- `errors.jsonl`: client-side errors when failures occur.

## Scoring

`scoring.yaml` is the scoring source of truth. The score chain is:

```text
metricResults score(0..1)
  -> dimensionScore(0..100)
  -> caseScore(0..100)
  -> totalScore(0..100)
```

Official score dimensions:

| Dimension | Metric codes |
| --- | --- |
| `outcome` | `reasoning.answer_accuracy`, `reasoning.task_completion_quality` |
| `execution` | `skill.execution_quality` |
| `orchestration` | `multi_agent.task_decompose_reasonability`, `multi_agent.collaboration_quality` |
| `efficiency` | `duration.normalized`, `token.normalized` |

The official summary fields are:

| Field | Meaning |
| --- | --- |
| `totalScore` | Final 0-100 score and primary ranking field. |
| `outcomeScore` | Result quality, answer correctness, business conclusion, and boundary handling. |
| `executionScore` | Tool, skill, data source, and evidence execution quality. |
| `orchestrationScore` | Task decomposition, multi-agent collaboration, and evidence organization. |
| `efficiencyScore` | Latency and token cost normalized into an efficiency score. |

All selected cases remain in the denominator. Runtime failures may still be
scored only when the service returns `scoreEligibility=SCORABLE`, useful final
output, and score-bearing metric results. The raw status remains visible through
`runtimeFailedCases`, `scorableNonSuccessCases`, and related diagnostic fields.

## Useful Commands

Run a full benchmark from a shell script:

```bash
#!/usr/bin/env bash
set -euo pipefail

./bin/agentx-bench run \
  --suite power_trade_standard_v1 \
  --method energent_swarm \
  --service-url "${AGENTX_SERVICE_URL:-http://127.0.0.1:8080}" \
  --team-name "${AGENTX_TEAM_NAME:-virtual-trader-team}" \
  --agent-mode multi_agent \
  --output "${AGENTX_OUTPUT:-evaluation_results/agentx}" \
  --timeout "${AGENTX_TIMEOUT:-7200}" \
  --request-timeout "${AGENTX_REQUEST_TIMEOUT:-120}" \
  --poll-interval "${AGENTX_POLL_INTERVAL:-30}" \
  --include-events \
  --include-raw-trace \
  --no-progress \
  --no-dashboard \
  --color never
```
