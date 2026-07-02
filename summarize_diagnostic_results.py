"""Build paper-facing diagnostic summaries from existing UGVU result JSON files.

This script is intentionally offline: it does not call any API. It reads the
current experiment result files, computes K=1 -> K=3 relative changes, and saves
one consolidated JSON summary for paper tables.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ugvu.diagnostics import attach_k_sweep_summary, finite_float

ROOT = Path("outputs")
SUMMARY_PATH = ROOT / "diagnostic_summary.json"

RESULT_FILES = {
    "cityscapes_open_class_api": ROOT / "cityscapes_api" / "cityscapes_api_results.json",
    "cityscapes_oracle_candidates": ROOT / "cityscapes_candidate" / "cityscapes_candidate_results.json",
    "ade20k_open_class_api": ROOT / "ade20k_api" / "ade20k_api_results.json",
    "ade20k_oracle_candidates": ROOT / "ade20k_candidate" / "candidate_results.json",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validation_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "validation" in payload:
        return payload["validation"]
    return []


def compact_table(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in records:
        metrics = item.get("metrics", {})
        rows.append(
            {
                "model": item.get("model"),
                "k": item.get("k"),
                "ok": item.get("ok", True),
                "mIoU": finite_float(metrics.get("mIoU")),
                "ECE": finite_float(metrics.get("ECE")),
                "Spearman": finite_float(metrics.get("Spearman")),
                "AUROC": finite_float(metrics.get("AUROC")),
                "error": item.get("error", ""),
            }
        )
    return rows


def main() -> None:
    summary: Dict[str, Any] = {
        "paper_framing": "Reliability and diagnostic probing of zero-shot dense spatial grounding, not SOTA supervised segmentation.",
        "protocols": {},
    }
    for name, path in RESULT_FILES.items():
        if not path.exists():
            summary["protocols"][name] = {"available": False, "path": str(path)}
            continue
        payload = load_json(path)
        records = validation_records(payload)
        summary["protocols"][name] = {
            "available": True,
            "path": str(path),
            "table": compact_table(records),
            "k_sweep_summary": attach_k_sweep_summary(records),
        }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

