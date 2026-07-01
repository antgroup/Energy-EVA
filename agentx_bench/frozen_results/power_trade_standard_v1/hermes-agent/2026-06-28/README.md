# Hermes Agent Power Trading Benchmark

This frozen package contains the Hermes Agent final baseline for AgentX Bench
`power_trade_standard_v1` on `2026-06-28`. The final score is computed from the
same three complete final batches: `final-batch-1`, `final-batch-2`, and
`final-batch-3`.

- Subject: `hermes-agent` / `Hermes Agent` / `hermes_agent` / `single_agent`
- Hermes package: `hermes-agent==0.17.0`
- Model: `gpt-5.5` via host `aiot-coding-maas.antdigital.com`
- Collection bounds: `1` attempt per case, `8` iterations per attempt, `600` seconds per case, `4` workers
- Final batches: `final-batch-1`, `final-batch-2`, `final-batch-3`
- Aggregate run id: `hermes-agent-power-trade-standard-v1-final`
- Total score mean: `40.1049`
- Score interval: `[39.3803, 40.9090]`
- Leaderboard eligible: `False`
- Failed cases across batches: `5.0`
- Completed cases across batches: `124.0`
- Incomplete cases across batches: `0.0`

Every retained final case row has a fresh Hermes session id, sanitized local
trajectory summary evidence, and score-bearing `metricResults`. Single-agent
orchestration metrics are marked `NOT_APPLICABLE`.
