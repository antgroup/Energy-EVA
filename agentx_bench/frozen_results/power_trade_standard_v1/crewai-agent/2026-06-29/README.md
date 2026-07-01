# CrewAI Agent Power Trading Benchmark

This frozen package contains the `CrewAI Agent` final baseline
for AgentX Bench `power_trade_standard_v1` on `2026-06-29`. Failed and timed-out
rows stay in the selected-case denominator.

- Subject: `crewai-agent` / `CrewAI Agent` / `crewai_agent` / `single_agent`
- Agent version: `crewai==1.15.1; upstream=6491f5a6639c49ed1835520e683a4c42c3eaf634`
- Model: `gpt-5.5` via host `aiot-coding-maas.antdigital.com`
- Collection bounds: `2` attempt per case, `600` seconds per case, `4` workers
- Final batches: `final-batch-1`, `final-batch-2`, `final-batch-3`
- Aggregate run id: `crewai-agent-power-trade-standard-v1-final`
- Total score mean: `53.9729`
- Score interval: `[53.8307, 54.0777]`
- Leaderboard eligible: `True`
- Failed cases across batches: `0.0`
- Completed cases across batches: `129.0`

Every retained final case row has a fresh session id, sanitized local trace or
failure evidence, and score-bearing `metricResults`. Single-agent orchestration metrics are marked `NOT_APPLICABLE`.
