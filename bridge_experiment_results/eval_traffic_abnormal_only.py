import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_LABELS = [
    "拥堵",
    "异常停车",
    "占道施工",
    "交通事故",
    "二轮闯入",
    "行人闯入",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate abnormal-only traffic results (multiclass + explanation quality)."
    )
    p.add_argument("--pred-jsonl", required=True, help="Path to prediction jsonl from batch_test_traffic.py")
    p.add_argument("--output-json", default="traffic_abnormal_eval.json", help="Path to summary json")
    p.add_argument("--error-csv", default="traffic_abnormal_errors.csv", help="Path to misclassification csv")
    p.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated abnormal labels",
    )
    p.add_argument("--pred-field", default="pred", help="Prediction text field")
    p.add_argument("--gt-field", default="gt", help="Ground-truth text field")
    p.add_argument(
        "--use-bertscore",
        action="store_true",
        help="Enable BERTScore (requires `pip install bert-score`)",
    )
    p.add_argument(
        "--bertscore-model",
        default="bert-base-chinese",
        help="Model name for bert_score",
    )
    return p.parse_args()


def norm_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


def tokenize_zh_char(s: str) -> List[str]:
    s = re.sub(r"\s+", "", s)
    return list(s)


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = dp[j] if dp[j] >= dp[j - 1] else dp[j - 1]
            prev = tmp
    return dp[m]


def rouge_l_f1(pred: str, gt: str) -> float:
    p = tokenize_zh_char(pred)
    g = tokenize_zh_char(gt)
    if not p or not g:
        return 0.0
    lcs = lcs_len(p, g)
    prec = lcs / len(p)
    rec = lcs / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def detect_label(text: str, labels: List[str]) -> Optional[str]:
    text = norm_text(text)
    if not text:
        return None
    # Exact first
    for lb in labels:
        if text == lb:
            return lb
    # Single hit
    hits = [lb for lb in labels if lb in text]
    if len(hits) == 1:
        return hits[0]
    # Regex extraction
    pats = [
        r"事故类型\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)",
        r"类型\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)",
        r"类别\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)",
    ]
    for pat in pats:
        m = re.search(pat, text)
        if not m:
            continue
        cand = m.group(1).strip()
        for lb in labels:
            if cand == lb or lb in cand:
                return lb
    if hits:
        return hits[0]
    return None


def get_gt_label(rec: Dict, labels: List[str], gt_field: str) -> Optional[str]:
    for key in ["gt_category", "label", "category", "gt_label"]:
        if key in rec:
            lb = detect_label(rec.get(key), labels)
            if lb:
                return lb
    if gt_field in rec:
        return detect_label(rec.get(gt_field), labels)
    return None


def field_extract(text: str) -> Dict[str, str]:
    text = norm_text(text)
    out = {"accident_type": "", "location": "", "lighting": "", "cause": "", "impact": ""}
    if not text:
        return out

    # accident_type
    m = re.search(r"(事故类型|类型|类别)\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)", text)
    if m:
        out["accident_type"] = m.group(2).strip()

    # location / lighting
    m = re.search(r"(地点|位置)\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)", text)
    if m:
        out["location"] = m.group(2).strip()
    m = re.search(r"(光照|天气|照明)\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)", text)
    if m:
        out["lighting"] = m.group(2).strip()

    # cause / impact by sentence hint
    for seg in re.split(r"[。；;]", text):
        seg = seg.strip()
        if not seg:
            continue
        if not out["cause"] and any(k in seg for k in ["原因", "由于", "因为", "引发", "导致"]):
            out["cause"] = seg
        if not out["impact"] and any(k in seg for k in ["影响", "后果", "造成", "风险", "拥堵"]):
            out["impact"] = seg

    return out


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def multiclass_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    supports = {l: 0 for l in labels}
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}

    for t, p in zip(y_true, y_pred):
        supports[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class = {}
    macro_f1 = 0.0
    total = len(y_true)
    acc = safe_div(sum(int(t == p) for t, p in zip(y_true, y_pred)), total)
    conf = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1

    for l in labels:
        p = safe_div(tp[l], tp[l] + fp[l])
        r = safe_div(tp[l], tp[l] + fn[l])
        f1 = safe_div(2 * p * r, p + r)
        macro_f1 += f1
        per_class[l] = {"precision": p, "recall": r, "f1": f1, "support": supports[l]}

    macro_f1 = safe_div(macro_f1, len(labels))
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": conf}


def maybe_bertscore(preds: List[str], gts: List[str], model_name: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(preds, gts, lang="zh", model_type=model_name, verbose=False)
        return float(f1.mean().item()), None
    except Exception as e:
        return None, str(e)


def main():
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if not labels:
        raise ValueError("labels 不能为空")

    rows = []
    with Path(args.pred_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    y_true, y_pred = [], []
    pred_texts, gt_texts = [], []
    unknown_gt = 0
    unknown_pred = 0
    errors = []

    field_keys = ["accident_type", "location", "lighting", "cause", "impact"]
    field_hit = {k: 0 for k in field_keys}
    field_cnt = {k: 0 for k in field_keys}
    rouge_scores = []

    for i, rec in enumerate(rows):
        gt_text = norm_text(rec.get(args.gt_field, ""))
        pred_text = norm_text(rec.get(args.pred_field, ""))
        gt_lb = get_gt_label(rec, labels, args.gt_field)
        pred_lb = detect_label(pred_text, labels)

        if gt_lb is None:
            unknown_gt += 1
            continue
        if pred_lb is None:
            unknown_pred += 1
            # Unknown prediction treated as wrong; map to first label not to crash confusion
            pred_lb = labels[0]

        y_true.append(gt_lb)
        y_pred.append(pred_lb)
        pred_texts.append(pred_text)
        gt_texts.append(gt_text)
        rouge_scores.append(rouge_l_f1(pred_text, gt_text))

        gt_f = field_extract(gt_text)
        pred_f = field_extract(pred_text)
        for k in field_keys:
            if gt_f[k]:
                field_cnt[k] += 1
                if pred_f[k] and (pred_f[k] in gt_f[k] or gt_f[k] in pred_f[k]):
                    field_hit[k] += 1

        if gt_lb != pred_lb:
            errors.append(
                {
                    "index": rec.get("index", i),
                    "id": rec.get("id", ""),
                    "video": rec.get("video", ""),
                    "gt_label": gt_lb,
                    "pred_label": pred_lb,
                    "pred_text": pred_text,
                }
            )

    cls_metrics = multiclass_metrics(y_true, y_pred, labels) if y_true else {}
    mean_rouge_l = safe_div(sum(rouge_scores), len(rouge_scores))

    bert_f1 = None
    bert_err = None
    if args.use_bertscore and pred_texts:
        bert_f1, bert_err = maybe_bertscore(pred_texts, gt_texts, args.bertscore_model)

    field_metrics = {}
    for k in field_keys:
        field_metrics[k] = {
            "hit": field_hit[k],
            "count_with_gt": field_cnt[k],
            "hit_rate": safe_div(field_hit[k], field_cnt[k]),
        }

    summary = {
        "input_file": args.pred_jsonl,
        "labels": labels,
        "total_records": len(rows),
        "valid_records_for_classification": len(y_true),
        "unknown_gt_count": unknown_gt,
        "unknown_pred_count": unknown_pred,
        "multiclass_detection": cls_metrics,
        "explanation_metrics": {
            "rouge_l_f1_mean": mean_rouge_l,
            "bertscore_f1_mean": bert_f1,
            "bertscore_error": bert_err,
            "field_hit_metrics": field_metrics,
        },
        "num_errors": len(errors),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    err_csv = Path(args.error_csv)
    err_csv.parent.mkdir(parents=True, exist_ok=True)
    with err_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "id", "video", "gt_label", "pred_label", "pred_text"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary: {out_json}")
    print(f"Saved errors : {err_csv}")


if __name__ == "__main__":
    main()

