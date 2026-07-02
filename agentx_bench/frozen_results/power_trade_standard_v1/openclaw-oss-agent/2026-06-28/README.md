# OpenClaw OSS Agent Frozen Benchmark Result

This directory contains the frozen AgentX Bench result for OpenClaw OSS Agent on `power_trade_standard_v1`.

Final batches are named `final-batch-1`, `final-batch-2`, and `final-batch-3`; internal collection attempt numbers are intentionally not part of the frozen result contract. Each final batch contains 43 successful case rows, 43 successful internal OpenClaw trace summaries, and per-case `metricResults` produced by `codex_manual_review_rubric_v1`.

The configured LLM judge endpoint was unreachable by TCP timeout during final scoring, so all three final batches were scored with one consistent manual-review rubric. Two case rows used fresh-session case retry because the original successful answer lacked retrievable internal trace; the retry audit is in `case_retry_audit.jsonl`.

Use `aggregate_summary.json` and `leaderboard_row.*` as the benchmark score source. The `manual_inputs/` directory is the frozen source data used by AgentX Bench manual import. Raw internal span-tree responses are not included; compact sanitized trace summaries and response hashes are retained.
