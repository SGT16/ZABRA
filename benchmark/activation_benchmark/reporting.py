from __future__ import annotations

import csv
import json
import os
import time
import uuid
from pathlib import Path

def replace_with_retry(source: Path, destination: Path, attempts: int = 8) -> None:
    """Atomic same-volume replacement, tolerating transient Windows scanners."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {5, 32, 33}):
                raise
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 2.0))


def atomic_torch_save(path: Path, payload) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def _with_derived_metrics(row: dict) -> dict:
    """Backfill error rate in both new and legacy fold rows."""
    item = dict(row)
    accuracy = item.get("test_accuracy", "")
    if accuracy not in {"", None, "nan"}:
        try:
            accuracy_value = float(accuracy)
            if item.get("test_error_rate", "") in {"", None, "nan"}:
                item["test_error_rate"] = 1.0 - accuracy_value
            if item.get("test_error_rate_percent", "") in {"", None, "nan"}:
                item["test_error_rate_percent"] = 100.0 * (1.0 - accuracy_value)
        except (TypeError, ValueError):
            pass
    return item


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict] = []
    old_fields: list[str] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f); old_fields = list(reader.fieldnames or []); old_rows = [_with_derived_metrics(item) for item in reader]
    row = _with_derived_metrics(row)
    key_fields = ("experiment_id", "protocol", "dataset", "model", "activation_scope", "activation_variant", "seed", "fold")
    key = tuple(str(row.get(field, "")) for field in key_fields)
    old_rows = [item for item in old_rows if tuple(str(item.get(field, "")) for field in key_fields) != key]
    old_rows.append(row)
    fields = list(row) + [field for field in old_fields if field not in row]
    temp = path.with_suffix(".tmp.csv")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(old_rows)
    replace_with_retry(temp, path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)
