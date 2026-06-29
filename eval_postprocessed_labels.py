import argparse
import json
from collections import Counter
from pathlib import Path


LABELS = [
    "\u62e5\u5835",
    "\u5f02\u5e38\u505c\u8f66",
    "\u5360\u9053\u65bd\u5de5",
    "\u591a\u8f66\u4e8b\u6545",
    "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165",
    "\u629b\u6d12\u7269",
]


def gt_from_video(video):
    if not isinstance(video, str):
        return None
    prefix = video.split("_")[0]
    return prefix if prefix in LABELS else None


def safe_div(a, b):
    return a / b if b else 0.0


def metrics(y_true, y_pred):
    labels = LABELS
    conf = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(y_true, y_pred):
        if g in conf and p in conf[g]:
            conf[g][p] += 1
    total = len(y_true)
    correct = sum(conf[l][l] for l in labels)
    per = {}
    macro = 0.0
    for l in labels:
        tp = conf[l][l]
        fp = sum(conf[g][l] for g in labels if g != l)
        fn = sum(conf[l][p] for p in labels if p != l)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = safe_div(2 * prec * rec, prec + rec)
        support = sum(conf[l].values())
        per[l] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        macro += f1
    return {
        "accuracy": safe_div(correct, total),
        "macro_f1": safe_div(macro, len(labels)),
        "per_class": per,
        "confusion_matrix": conf,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate pre-extracted postprocessed labels.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--label-field", default="pred_post_label")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    y_true = []
    y_pred = []
    unknown_gt = 0
    unknown_pred = 0
    pred_counter = Counter()
    reason_counter = Counter()
    total = 0
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            gt = gt_from_video(rec.get("video", ""))
            pred = rec.get(args.label_field)
            if gt is None:
                unknown_gt += 1
                continue
            if pred not in LABELS:
                unknown_pred += 1
                continue
            y_true.append(gt)
            y_pred.append(pred)
            pred_counter[pred] += 1
            if rec.get("pred_postprocess_reason"):
                reason_counter[rec.get("pred_postprocess_reason")] += 1

    out = {
        "input_file": args.input,
        "labels": LABELS,
        "total_records": total,
        "valid_records_for_classification": len(y_true),
        "unknown_gt_count": unknown_gt,
        "unknown_pred_count": unknown_pred,
        "pred_counter": dict(pred_counter),
        "postprocess_reason_counter": dict(reason_counter),
        "multiclass_detection": metrics(y_true, y_pred),
    }
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("saved =", args.output_json)


if __name__ == "__main__":
    main()
