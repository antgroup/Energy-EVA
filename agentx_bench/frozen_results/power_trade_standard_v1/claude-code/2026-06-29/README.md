# Claude Code Power Trading Benchmark

This frozen package contains the Claude Code final baseline for AgentX Bench
`power_trade_standard_v1` on `2026-06-29`. Failed and timed-out rows stay in the
selected-case denominator.

- Subject: `claude-code` / `Claude Code` / `claude_code` / `single_agent`
- Claude Code version: `2.1.177 (Claude Code)`
- Requested model: `sonnet`
- Tools: `WebSearch, WebFetch`
- Permission mode: `bypassPermissions`
- Effort: `low`
- Collection bounds: `1` attempt per case, `600` seconds per case, `2` workers
- Final batches: `final-batch-1`, `final-batch-2`, `final-batch-3`
- Aggregate run id: `claude-code-power-trade-standard-v1-final`
- Total score mean: `42.1808`
- Score interval: `[41.7902, 42.6200]`
- Leaderboard eligible: `False`
- Failed cases across batches: `1.0`
- Completed cases across batches: `128.0`

Every retained final case row has a fresh Claude Code session id, sanitized
stream-json trajectory or failure evidence, and score-bearing `metricResults`.
Single-agent orchestration metrics are marked `NOT_APPLICABLE`.
