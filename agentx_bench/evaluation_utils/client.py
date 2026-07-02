"""HTTP client for the AgentX Evaluation Service API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class AgentXServiceError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, payload: Any | None = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class AgentXClient:
    def __init__(self, base_url: str, timeout: float = 30.0, headers: dict[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = dict(headers or {})

    def submit_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/agentx/evaluation-runs", payload)

    def get_status(self, eval_run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/agentx/evaluation-runs/{eval_run_id}", None)

    def get_results(self, eval_run_id: str, include_events: bool = False,
                    include_metric_results: bool = True, include_raw_trace: bool = False) -> dict[str, Any]:
        query = (
            f"?includeEvents={str(include_events).lower()}"
            f"&includeMetricResults={str(include_metric_results).lower()}"
            f"&includeRawTrace={str(include_raw_trace).lower()}"
        )
        return self._request("GET", f"/api/agentx/evaluation-runs/{eval_run_id}/results{query}", None)

    def get_console_suite(self, suite_key: str) -> dict[str, Any]:
        return self._request("GET", f"/api/evaluation/console/suites/{suite_key}", None)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json", **self.headers}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return _decode_json(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read()
            payload_value = _decode_json(body, allow_empty=True)
            message = _extract_error_message(payload_value) or f"HTTP {exc.code} from AgentX service"
            raise AgentXServiceError(message, status=exc.code, payload=payload_value) from exc
        except urllib.error.URLError as exc:
            raise AgentXServiceError(f"AgentX service request failed: {exc.reason}") from exc


def parse_headers(values: list[str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values or []:
        if ":" in value:
            key, header_value = value.split(":", 1)
        elif "=" in value:
            key, header_value = value.split("=", 1)
        else:
            raise ValueError(f"invalid header format: {value}")
        key = key.strip()
        header_value = header_value.strip()
        if not key or not header_value:
            raise ValueError(f"invalid header format: {value}")
        headers[key] = header_value
    return headers


def _decode_json(raw: bytes, allow_empty: bool = False) -> dict[str, Any]:
    if not raw:
        if allow_empty:
            return {}
        raise AgentXServiceError("empty JSON response from AgentX service")
    try:
        value = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise AgentXServiceError(f"invalid JSON response from AgentX service: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentXServiceError("AgentX service response must be a JSON object")
    return value


def _extract_error_message(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("errorMessage", "message", "error"):
            if payload.get(key):
                return str(payload[key])
    return None
