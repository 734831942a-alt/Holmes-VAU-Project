import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_LABELS = "多车事故,拥堵,异常停车,占道施工,二轮车辆闯入,抛洒物"


def parse_csv(text, cast=str):
    vals = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            vals.append(cast(part))
    return vals


def metric_from_eval(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    mc = data["multiclass_detection"]
    per = mc["per_class"]
    return {
        "accuracy": mc["accuracy"],
        "macro_f1": mc["macro_f1"],
        "multicar_precision": per["多车事故"]["precision"],
        "multicar_recall": per["多车事故"]["recall"],
        "multicar_f1": per["多车事故"]["f1"],
        "parking_f1": per["异常停车"]["f1"],
        "congestion_f1": per["拥堵"]["f1"],
        "num_errors": data.get("num_errors"),
    }


def build_grid(args):
    keys = [
        ("allowed_base_labels", parse_csv(args.allowed_base_labels_grid)),
        ("score_thr", parse_csv(args.score_thr_grid, float)),
        ("lateral_dx", parse_csv(args.lateral_dx_grid, float)),
        ("lateral_dy", parse_csv(args.lateral_dy_grid, float)),
        ("compactness", parse_csv(args.compactness_grid, float)),
    ]
    for values in itertools.product(*[v for _, v in keys]):
        yield dict(zip([k for k, _ in keys], values))


def run_cmd(cmd, dry_run=False):
    print("+", " ".join(map(str, cmd)))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description="Sweep LCRM-lite thresholds on a validation split. Test set should not be used here."
    )
    ap.add_argument("--base-pred", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--postprocess-script", default="postprocess_lcrm_lite_fusion.py")
    ap.add_argument(
        "--eval-script",
        default="eval_traffic_fixed.py",
        help="Evaluation script. Defaults to the fixed v1 protocol.",
    )
    ap.add_argument(
        "--base-pred-field",
        default="pred_label_only",
        help="Prediction field used by postprocess_lcrm_lite_fusion.py. Use pred_label_only for clean validation sweeps.",
    )
    ap.add_argument(
        "--fused-field",
        default="pred_lcrm_lite_fused",
        help="Fused prediction field written by postprocess_lcrm_lite_fusion.py and read by eval.",
    )
    ap.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="Fixed label list passed to postprocess for compatibility validation.",
    )
    ap.add_argument(
        "--gt-field",
        default="gt_label",
        help="GT field passed to eval_traffic_fixed.py. The fixed evaluator still prioritizes video prefix.",
    )
    ap.add_argument("--source", default="lcrm", choices=["auto", "lcrm", "vscm_debug"])
    ap.add_argument(
        "--allowed-base-labels-grid",
        default="异常停车;异常停车,拥堵",
        help='Semicolon-separated grid items. Example: "异常停车;异常停车,拥堵".',
    )
    ap.add_argument("--score-thr-grid", default="0.60,0.65,0.70")
    ap.add_argument("--lateral-dx-grid", default="0.35,0.40,0.45")
    ap.add_argument("--lateral-dy-grid", default="0.15,0.20,0.25")
    ap.add_argument("--compactness-grid", default="0.30,0.35,0.40")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    allowed_items = [x.strip() for x in args.allowed_base_labels_grid.split(";") if x.strip()]

    grid = []
    for allowed in allowed_items:
        for score, dx, dy, compact in itertools.product(
            parse_csv(args.score_thr_grid, float),
            parse_csv(args.lateral_dx_grid, float),
            parse_csv(args.lateral_dy_grid, float),
            parse_csv(args.compactness_grid, float),
        ):
            grid.append({
                "allowed_base_labels": allowed,
                "score_thr": score,
                "lateral_dx": dx,
                "lateral_dy": dy,
                "compactness": compact,
            })

    for i, cfg in enumerate(grid, 1):
        tag = (
            f"run{i:03d}_base-{cfg['allowed_base_labels'].replace(',', '+')}"
            f"_s{cfg['score_thr']:.2f}_dx{cfg['lateral_dx']:.2f}"
            f"_dy{cfg['lateral_dy']:.2f}_cmp{cfg['compactness']:.2f}"
        )
        pred_path = out_dir / f"{tag}.jsonl"
        eval_path = out_dir / f"{tag}_eval.json"
        err_path = out_dir / f"{tag}_errors.csv"

        post_cmd = [
            sys.executable, args.postprocess_script,
            "--base-pred", args.base_pred,
            "--base-pred-field", args.base_pred_field,
            "--ovd", args.ovd,
            "--output", str(pred_path),
            "--fused-field", args.fused_field,
            "--labels", args.labels,
            "--source", args.source,
            "--allowed-base-labels", cfg["allowed_base_labels"],
            "--score-thr", str(cfg["score_thr"]),
            "--lateral-suppress-dx", str(cfg["lateral_dx"]),
            "--lateral-suppress-dy", str(cfg["lateral_dy"]),
            "--min-pair-compactness", str(cfg["compactness"]),
        ]
        eval_cmd = [
            sys.executable, args.eval_script,
            "--pred-jsonl", str(pred_path),
            "--output-json", str(eval_path),
            "--error-csv", str(err_path),
            "--pred-field", args.fused_field,
            "--gt-field", args.gt_field,
        ]

        run_cmd(post_cmd, args.dry_run)
        run_cmd(eval_cmd, args.dry_run)
        if args.dry_run:
            continue

        row = {"tag": tag, **cfg, **metric_from_eval(eval_path)}
        results.append(row)
        print("RESULT", json.dumps(row, ensure_ascii=False))

    if args.dry_run:
        print("dry_run grid size:", len(grid))
        return

    results.sort(key=lambda x: (x["macro_f1"], x["multicar_f1"]), reverse=True)
    summary_json = out_dir / "sweep_summary.json"
    summary_csv = out_dir / "sweep_summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["tag"])
        writer.writeheader()
        writer.writerows(results)

    print("\n=== top results ===")
    for row in results[: args.top_k]:
        print(json.dumps(row, ensure_ascii=False))
    print("summary_json:", summary_json)
    print("summary_csv:", summary_csv)


if __name__ == "__main__":
    main()
