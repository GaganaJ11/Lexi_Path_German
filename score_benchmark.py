import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

LEVELS = ["A1", "A2", "B1"]
LEVEL_RANK = {"Pre-A1": 0, "A1": 1, "A2": 2, "B1": 3}


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def successful(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("success")]


def retrieval_coverage(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    coverages: List[float] = []
    avg_chunks: List[float] = []
    fallback_rates: List[float] = []

    for row in rows:
        stats = row.get("retrieval_stats", {}) or {}
        turns = stats.get("turns", 0) or 0
        turns_with_chunks = stats.get("turns_with_chunks", 0) or 0
        total_chunks = stats.get("total_chunks", 0) or 0
        fallback_turns = stats.get("fallback_turns", 0) or 0
        if turns > 0:
            coverages.append(safe_div(turns_with_chunks, turns))
            avg_chunks.append(safe_div(total_chunks, turns))
            fallback_rates.append(safe_div(fallback_turns, turns))

    return {
        "retrieval_turn_coverage": round(average(coverages), 4),
        "retrieval_avg_chunks_per_turn": round(average(avg_chunks), 4),
        "retrieval_fallback_rate": round(average(fallback_rates), 4),
    }


def qualitative_memory_recall(avg_keyword_hits: float, recall_rate: float) -> str:
    if recall_rate <= 0 or avg_keyword_hits <= 0:
        return "Weak"
    if avg_keyword_hits >= 3:
        return "Strong"
    if avg_keyword_hits >= 2:
        return "Moderate"
    return "Weak"


def trace_levels(row: Dict[str, Any]) -> List[str]:
    trace = row.get("diagnostic_trace", [])
    out: List[str] = []
    if not isinstance(trace, list):
        return out
    for item in trace:
        level = item.get("target_level") or item.get("question_level") or ""
        if isinstance(level, str) and level:
            out.append(level)
    return out


def diagnosis_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = successful(rows)
    y_true = [row.get("gold_level", "") for row in ok]
    y_pred = [row.get("detected_level", "") for row in ok]
    diag_turns = [len(row.get("diagnostic_trace", [])) for row in ok]
    over = 0
    under = 0
    for gold, pred in zip(y_true, y_pred):
        if gold not in LEVEL_RANK or pred not in LEVEL_RANK:
            continue
        if LEVEL_RANK[pred] > LEVEL_RANK[gold]:
            over += 1
        elif LEVEL_RANK[pred] < LEVEL_RANK[gold]:
            under += 1

    level_spans = []
    b1_probe_hits = 0
    for row in ok:
        ranks = [LEVEL_RANK[level] for level in trace_levels(row) if level in LEVEL_RANK]
        if ranks:
            level_spans.append(max(ranks) - min(ranks))
        if "B1" in set(trace_levels(row)):
            b1_probe_hits += 1

    return {
        "runs": len(rows),
        "run_success_rate": round(safe_div(len(ok), len(rows)), 4),
        "diagnostic_level_accuracy": round(safe_div(sum(1 for g, p in zip(y_true, y_pred) if g == p), len(y_true)), 4),
        "diagnostic_macro_f1": round(compute_macro_f1(y_true, y_pred), 4),
        "diagnostic_overplacement_rate": round(safe_div(over, len(y_true)), 4),
        "diagnostic_underplacement_rate": round(safe_div(under, len(y_true)), 4),
        "diagnostic_turns_avg": round(average(diag_turns), 4),
        "diagnostic_level_span_avg": round(average(level_spans), 4),
        "b1_probe_rate": round(safe_div(b1_probe_hits, len(ok)), 4),
    }


def tutoring_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = successful(rows)
    metrics = [row.get("metrics", {}) for row in ok]
    return {
        "runs": len(rows),
        "run_success_rate": round(safe_div(len(ok), len(rows)), 4),
        "assistant_turns_avg": round(average([metric.get("assistant_turns", 0.0) for metric in metrics]), 4),
        "avg_response_chars": round(average([metric.get("avg_response_chars", 0.0) for metric in metrics]), 4),
        "followup_question_rate": round(average([metric.get("followup_question_rate", 0.0) for metric in metrics]), 4),
        **retrieval_coverage(ok),
    }


def continuity_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = successful(rows)
    metrics = [row.get("metrics", {}) for row in ok]
    recall_rate = safe_div(
        sum(1 for metric in metrics if metric.get("session2_memory_recall", False)),
        len(metrics),
    )
    keyword_hits_avg = average([metric.get("session2_memory_keyword_hits", 0.0) for metric in metrics])
    return {
        "runs": len(rows),
        "run_success_rate": round(safe_div(len(ok), len(rows)), 4),
        "session2_memory_recall_rate": round(recall_rate, 4),
        "session2_memory_keyword_hits_avg": round(keyword_hits_avg, 4),
        "multi_session_recall": qualitative_memory_recall(keyword_hits_avg, recall_rate),
        "assistant_turns_avg": round(average([metric.get("assistant_turns", 0.0) for metric in metrics]), 4),
        "avg_response_chars": round(average([metric.get("avg_response_chars", 0.0) for metric in metrics]), 4),
        "followup_question_rate": round(average([metric.get("followup_question_rate", 0.0) for metric in metrics]), 4),
        **retrieval_coverage(ok),
    }


def score_report(report: Dict[str, Any]) -> Dict[str, Any]:
    by_system_and_suite: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in report.get("results", []):
        by_system_and_suite[row.get("system", "unknown")][row.get("suite", "unknown")].append(row)

    summary: Dict[str, Any] = {
        "source_report": report.get("_source_path", ""),
        "generated_at_epoch": report.get("generated_at_epoch"),
        "system_summary": {},
        "flat_rows": [],
    }

    for system, suites in sorted(by_system_and_suite.items()):
        diagnosis = diagnosis_summary(suites.get("diagnosis", []))
        tutoring = tutoring_summary(suites.get("tutoring", []))
        continuity = continuity_summary(suites.get("continuity", []))
        summary["system_summary"][system] = {
            "diagnosis": diagnosis,
            "tutoring": tutoring,
            "continuity": continuity,
        }

        flat_row = {"system": system}
        flat_row.update({f"diagnosis_{k}": v for k, v in diagnosis.items()})
        flat_row.update({f"tutoring_{k}": v for k, v in tutoring.items()})
        flat_row.update({f"continuity_{k}": v for k, v in continuity.items()})
        summary["flat_rows"].append(flat_row)

    return summary


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
    parser = argparse.ArgumentParser(description="Score benchmark evaluation report")
    parser.add_argument("--report", required=True, help="Path to benchmark_report.json")
    parser.add_argument("--out-json", default="reports/benchmark_summary.json", help="Output JSON summary")
    parser.add_argument("--out-csv", default="reports/benchmark_table.csv", help="Output CSV summary")
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
    write_csv(scored["flat_rows"], out_csv)

    print("Benchmark scoring complete.")
    print(f"JSON: {out_json.resolve()}")
    print(f"CSV : {out_csv.resolve()}")


if __name__ == "__main__":
    main()
