import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

LEVELS = ["A1", "A2", "B1"]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0

    f1s: List[float] = []
    for level in LEVELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == level and p == level)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != level and p == level)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == level and p != level)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision == 0 and recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))

    return sum(f1s) / len(f1s)


def memory_recall_rate(result_rows: List[Dict[str, Any]]) -> float:
    if not result_rows:
        return 0.0
    hits = sum(1 for row in result_rows if row.get("metrics", {}).get("session2_memory_recall", False))
    return safe_div(hits, len(result_rows))


def avg_metric(result_rows: List[Dict[str, Any]], metric_key: str) -> float:
    vals = [row.get("metrics", {}).get(metric_key, 0.0) for row in result_rows]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def retrieval_coverage(result_rows: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    coverages: List[float] = []
    avg_chunks: List[float] = []
    fallback_rates: List[float] = []

    for row in result_rows:
        stats = row.get("retrieval_stats", {}) or {}
        turns = stats.get("turns", 0) or 0
        turns_with_chunks = stats.get("turns_with_chunks", 0) or 0
        total_chunks = stats.get("total_chunks", 0) or 0
        fallback_turns = stats.get("fallback_turns", 0) or 0

        if turns > 0:
            coverages.append(safe_div(turns_with_chunks, turns))
            avg_chunks.append(safe_div(total_chunks, turns))
            fallback_rates.append(safe_div(fallback_turns, turns))

    if not coverages:
        return 0.0, 0.0, 0.0

    return (
        sum(coverages) / len(coverages),
        sum(avg_chunks) / len(avg_chunks),
        sum(fallback_rates) / len(fallback_rates),
    )


def score_report(report: Dict[str, Any]) -> Dict[str, Any]:
    scenario = report.get("scenario", {})
    gold_level = scenario.get("gold_level")

    by_system: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in report.get("results", []):
        by_system[row.get("system", "unknown")].append(row)

    system_rows: List[Dict[str, Any]] = []

    for system_name, rows in by_system.items():
        successful = [r for r in rows if r.get("success")]
        success_rate = safe_div(len(successful), len(rows))

        y_true: List[str] = []
        y_pred: List[str] = []
        if gold_level:
            for r in successful:
                y_true.append(gold_level)
                y_pred.append(r.get("detected_level", ""))

        level_accuracy = safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p), len(y_true)) if y_true else 0.0
        macro_f1 = compute_macro_f1(y_true, y_pred) if y_true else 0.0

        retrieval_cov, retrieval_avg_chunks, retrieval_fallback = retrieval_coverage(successful)

        system_rows.append(
            {
                "system": system_name,
                "runs": len(rows),
                "run_success_rate": round(success_rate, 4),
                "diagnostic_level_accuracy": round(level_accuracy, 4),
                "diagnostic_macro_f1": round(macro_f1, 4),
                "assistant_turns_avg": round(avg_metric(successful, "assistant_turns"), 4),
                "avg_response_chars": round(avg_metric(successful, "avg_response_chars"), 4),
                "followup_question_rate": round(avg_metric(successful, "followup_question_rate"), 4),
                "session2_memory_recall_rate": round(memory_recall_rate(successful), 4),
                "session2_memory_keyword_hits_avg": round(avg_metric(successful, "session2_memory_keyword_hits"), 4),
                "retrieval_turn_coverage": round(retrieval_cov, 4),
                "retrieval_avg_chunks_per_turn": round(retrieval_avg_chunks, 4),
                "retrieval_fallback_rate": round(retrieval_fallback, 4),
            }
        )

    return {
        "source_report": report.get("_source_path", ""),
        "scenario_name": scenario.get("name", "unknown"),
        "gold_level": gold_level,
        "system_metrics": sorted(system_rows, key=lambda x: x["system"]),
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score evaluation report into paper-ready metrics")
    parser.add_argument("--report", required=True, help="Path to evaluation_report.json")
    parser.add_argument("--out-json", default="reports/metrics_summary.json", help="Output JSON summary")
    parser.add_argument("--out-csv", default="reports/metrics_table.csv", help="Output CSV metrics table")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["_source_path"] = str(report_path.resolve())

    scored = score_report(report)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(scored["system_metrics"], out_csv)

    print("Scoring complete.")
    print(f"JSON: {out_json.resolve()}")
    print(f"CSV : {out_csv.resolve()}")


if __name__ == "__main__":
    main()
