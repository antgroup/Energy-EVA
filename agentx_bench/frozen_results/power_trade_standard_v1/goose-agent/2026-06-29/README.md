# Goose Agent Power Trading Benchmark

This frozen package contains the `Goose Agent` final baseline
for AgentX Bench `power_trade_standard_v1` on `2026-06-29`. Failed and timed-out
rows stay in the selected-case denominator.

- Subject: `goose-agent` / `Goose Agent` / `goose_agent` / `single_agent`
- Agent version: `1.39.0; upstream=2cc1140dc1e8d8dc8576bb41bdd5f9c5631a36d6`
- Model: `gpt-5.5` via host `aiot-coding-maas.antdigital.com`
- Collection bounds: `1` attempt per case, `600` seconds per case, `4` workers
- Final batches: `final-batch-1`, `final-batch-2`, `final-batch-3`
- Aggregate run id: `goose-agent-power-trade-standard-v1-final`
- Total score mean: `33.1491`
- Score interval: `[32.9373, 33.3654]`
- Leaderboard eligible: `True`
- Failed cases across batches: `0.0`
- Completed cases across batches: `129.0`

Every retained final case row has a fresh session id, sanitized local trace or
failure evidence, and score-bearing `metricResults`. Single-agent orchestration metrics are marked `NOT_APPLICABLE`.
