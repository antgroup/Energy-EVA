"""Artifact writing helpers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .models import RunContext


class ArtifactWriter:
    def __init__(self, context: RunContext):
        self.context = context
        self.run_dir = (
            context.output_root
            / _slug(context.suite.suite_id)
            / _slug(context.subject.subject_id)
            / _slug(context.run_id)
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "errors.jsonl").touch(exist_ok=True)

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def _prepare_path(self, name: str) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.path(name)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self._prepare_path(name)
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self._prepare_path(name)
        path.write_text(text, encoding="utf-8")
        return path

    def append_error(self, payload: dict[str, Any]) -> None:
        with self._prepare_path("errors.jsonl").open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            file_obj.write("\n")

    def append_status(self, payload: dict[str, Any]) -> None:
        self.write_json("run_status.json", payload)
        with self._prepare_path("status_history.jsonl").open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            file_obj.write("\n")

    def write_jsonl(self, name: str, rows: Iterable[dict[str, Any]]) -> Path:
        path = self._prepare_path(name)
        with path.open("w", encoding="utf-8") as file_obj:
            for row in rows:
                file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                file_obj.write("\n")
        return path

    def write_csv(self, name: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
        path = self._prepare_path(name)
        if fieldnames is None:
            keys: list[str] = []
            for row in rows:
                for key in row.keys():
                    if key not in keys:
                        keys.append(key)
            fieldnames = keys
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
        return path


def write_root_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")


def write_root_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _slug(value: str) -> str:
    value = str(value or "unknown").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:120] or "unknown"
