import argparse
import csv
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path


TARGET = "\u591a\u8f66\u4e8b\u6545"


def safe_div(a, b):
    return a / b if b else 0.0


def has_any(text, words):
    return any(w in text for w in words)


def count_any(text, words):
    return sum(1 for w in words if w in text)


def suggest_label_for_false_multicar(pred_text):
    """
    Conservative relabeling for official false positives to multi-car accident.

    This function is only called when official eval already says:
        gt_label != 多车事故 and pred_label == 多车事故
    It returns None when evidence is not strong enough.
    """
    text = "" if pred_text is None else str(pred_text)

    collision_words = ["\u78b0\u649e", "\u8ffd\u5c3e", "\u5250\u8e6d", "\u649e\u51fb", "\u4e24\u8f66\u76f8\u649e"]
    accident_scene_words = ["\u4e8b\u6545\u73b0\u573a", "\u4ea4\u8b66", "\u5904\u7f6e", "\u591a\u8f66\u4e8b\u6545"]
    if has_any(text, collision_words) or count_any(text, accident_scene_words) >= 2:
        return None, ""

    two_wheel_words = ["\u4e8c\u8f6e", "\u6469\u6258", "\u7535\u52a8\u8f66", "\u975e\u673a\u52a8\u8f66", "\u5934\u76d4"]
    if count_any(text, two_wheel_words) >= 2:
        return "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165", "strong_two_wheel_text"

    debris_words = ["\u629b\u6d12", "\u629b\u6492", "\u6563\u843d", "\u5f02\u7269", "\u8def\u9762\u969c\u788d", "\u969c\u788d\u7269"]
    if count_any(text, debris_words) >= 2:
        return "\u629b\u6d12\u7269", "strong_debris_text"

    construction_words = ["\u5360\u9053\u65bd\u5de5", "\u65bd\u5de5", "\u9525\u6876", "\u56f4\u6321", "\u4f5c\u4e1a\u8f66", "\u5de5\u7a0b\u8f66"]
    if count_any(text, construction_words) >= 2:
        return "\u5360\u9053\u65bd\u5de5", "strong_construction_text"

    single_stop = re.search(r"(\u4e00\u8f86|\u5355\u8f66).{0,30}(\u505c\u9760|\u505c\u5728|\u9759\u6b62|\u505c\u7559)", text)
    parking_words = ["\u5f02\u5e38\u505c\u8f66", "\u505c\u9760", "\u505c\u5728", "\u9759\u6b62", "\u53cc\u95ea", "\u5e94\u6025\u8f66\u9053", "\u6545\u969c", "\u4e0b\u8f66\u67e5\u770b"]
    if single_stop or count_any(text, parking_words) >= 4:
        return "\u5f02\u5e38\u505c\u8f66", "strong_parking_text"

    congestion_words = ["\u62e5\u5835", "\u6392\u961f", "\u7f13\u884c", "\u4f4e\u901f", "\u8f66\u6d41\u5bc6\u96c6", "\u8f66\u8f86\u5bc6\u96c6", "\u6574\u4f53", "\u5927\u8303\u56f4"]
    if count_any(text, congestion_words) >= 4:
        return "\u62e5\u5835", "strong_congestion_text"

    return None, ""


def recompute_metrics(conf, labels):
    total = sum(sum(row.values()) for row in conf.values())
    correct = sum(conf[l][l] for l in labels)
    per_class = {}
    macro = 0.0
    for l in labels:
        tp = conf[l][l]
        fp = sum(conf[g][l] for g in labels if g != l)
        fn = sum(conf[l][p] for p in labels if p != l)
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f1 = safe_div(2 * p * r, p + r)
        per_class[l] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": sum(conf[l].values()),
        }
        macro += f1
    return {
        "accuracy": safe_div(correct, total),
        "macro_f1": safe_div(macro, len(labels)),
        "per_class": per_class,
        "confusion_matrix": conf,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Postprocess only official false positives to multi-car using eval error CSV."
    )
    ap.add_argument("--eval-json", required=True, help="Original E8 eval json")
    ap.add_argument("--error-csv", required=True, help="Original E8 eval error csv")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    base = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    labels = base["labels"]
    conf = deepcopy(base["multiclass_detection"]["confusion_matrix"])

    changes = []
    reasons = Counter()
    with Path(args.error_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gt = row.get("gt_label", "")
            pred = row.get("pred_label", "")
            if gt == TARGET or pred != TARGET:
                continue
            new_label, reason = suggest_label_for_false_multicar(row.get("pred_text", ""))
            if not new_label or new_label not in labels or new_label == TARGET:
                continue
            if conf[gt][TARGET] <= 0:
                continue
            conf[gt][TARGET] -= 1
            conf[gt][new_label] += 1
            reasons[reason] += 1
            changes.append(
                {
                    "video": row.get("video", ""),
                    "gt_label": gt,
                    "old_pred_label": TARGET,
                    "new_pred_label": new_label,
                    "reason": reason,
                }
            )

    out = deepcopy(base)
    out["postprocess"] = {
        "method": "official_false_positive_to_multicar_only",
        "changed": len(changes),
        "reason_counter": dict(reasons),
        "changes": changes,
    }
    out["multiclass_detection"] = recompute_metrics(conf, labels)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("changed =", len(changes))
    print("reasons =", dict(reasons))
    m = out["multiclass_detection"]
    print("accuracy =", m["accuracy"])
    print("macro_f1 =", m["macro_f1"])
    target = TARGET
    fp = sum(conf[g].get(target, 0) for g in labels if g != target)
    fn = sum(conf[target].get(p, 0) for p in labels if p != target)
    print("FP_to_multicar =", fp)
    print("FN_multicar =", fn)
    print("saved =", args.output_json)


if __name__ == "__main__":
    main()
