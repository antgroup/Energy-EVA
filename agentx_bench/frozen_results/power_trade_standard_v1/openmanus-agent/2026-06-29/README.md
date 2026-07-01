# OpenManus Agent Power Trading Benchmark

This frozen package contains the `OpenManus Agent` final baseline
for AgentX Bench `power_trade_standard_v1` on `2026-06-29`. Failed and timed-out
rows stay in the selected-case denominator.

- Subject: `openmanus-agent` / `OpenManus Agent` / `openmanus_agent` / `single_agent`
- Agent version: `OpenManus upstream=52a13f2a57d8c7f6737eefb02ccf569594d44273`
- Model: `gpt-5.5` via host `aiot-coding-maas.antdigital.com`
- Collection bounds: `2` attempt per case, `600` seconds per case, `7` workers
- Final batches: `final-batch-1`, `final-batch-2`, `final-batch-3`
- Aggregate run id: `openmanus-agent-power-trade-standard-v1-final`
- Total score mean: `34.8078`
- Score interval: `[34.4889, 35.2167]`
- Leaderboard eligible: `True`
- Failed cases across batches: `0.0`
- Completed cases across batches: `129.0`

Every retained final case row has a fresh session id, sanitized local trace or
failure evidence, and score-bearing `metricResults`. Single-agent orchestration metrics are marked `NOT_APPLICABLE`.
