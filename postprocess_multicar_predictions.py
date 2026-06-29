import argparse
import json
import re
from collections import Counter
from pathlib import Path


TARGET = "\u591a\u8f66\u4e8b\u6545"


def norm(text):
    return "" if text is None else str(text)


def has_any(text, words):
    return any(w in text for w in words)


def count_any(text, words):
    return sum(1 for w in words if w in text)


def fix_multicar_label(pred_text):
    """
    Conservative post-processing for E8 only.

    Only rewrite when the original extracted label is multi-car accident
    but the explanation text contains stronger evidence for another class.
    """
    text = norm(pred_text)

    two_wheel_words = [
        "\u4e8c\u8f6e", "\u6469\u6258", "\u7535\u52a8\u8f66", "\u975e\u673a\u52a8\u8f66", "\u95ef\u5165\u9ad8\u901f",
        "\u9a91\u884c", "\u5934\u76d4",
    ]
    parking_words = [
        "\u5355\u8f66", "\u4e00\u8f86", "\u72ec\u81ea", "\u505c\u9760", "\u505c\u5728", "\u9759\u6b62",
        "\u53cc\u95ea", "\u5e94\u6025\u8f66\u9053", "\u4e0b\u8f66\u67e5\u770b", "\u6545\u969c",
    ]
    congestion_words = [
        "\u62e5\u5835", "\u6392\u961f", "\u7f13\u884c", "\u4f4e\u901f", "\u8f66\u6d41\u5bc6\u96c6",
        "\u8f66\u8f86\u5bc6\u96c6", "\u505c\u6ede\u4e0d\u524d", "\u6574\u4f53", "\u5927\u8303\u56f4",
    ]
    debris_words = [
        "\u629b\u6d12", "\u629b\u6492", "\u6563\u843d", "\u5f02\u7269", "\u969c\u788d\u7269", "\u8def\u9762\u969c\u788d",
    ]
    construction_words = [
        "\u65bd\u5de5", "\u5360\u9053\u65bd\u5de5", "\u9525\u6876", "\u56f4\u6321", "\u4f5c\u4e1a\u8f66", "\u5de5\u7a0b\u8f66",
    ]
    accident_words = [
        "\u78b0\u649e", "\u8ffd\u5c3e", "\u5250\u8e6d", "\u4e8b\u6545", "\u4e24\u8f66", "\u4e24\u8f86",
        "\u591a\u8f66", "\u591a\u8f86", "\u649e", "\u4e8b\u6545\u73b0\u573a",
    ]

    # Strong non-accident classes first.
    if has_any(text, two_wheel_words) and not has_any(text, ["\u78b0\u649e", "\u8ffd\u5c3e", "\u5250\u8e6d"]):
        return "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165", "two_wheel_keywords"

    if has_any(text, debris_words) and not has_any(text, ["\u78b0\u649e", "\u8ffd\u5c3e"]):
        return "\u629b\u6d12\u7269", "debris_keywords"

    if has_any(text, construction_words) and not has_any(text, ["\u78b0\u649e", "\u8ffd\u5c3e"]):
        return "\u5360\u9053\u65bd\u5de5", "construction_keywords"

    accident_score = count_any(text, accident_words)
    parking_score = count_any(text, parking_words)
    congestion_score = count_any(text, congestion_words)

    # If the text itself describes a single stopped vehicle and lacks concrete collision evidence,
    # multi-car is likely a hallucinated label.
    single_vehicle_pattern = re.search(r"(\u4e00\u8f86|\u5355\u8f66).{0,20}(\u505c|\u9759\u6b62|\u505c\u9760)", text)
    if single_vehicle_pattern and accident_score <= 1:
        return "\u5f02\u5e38\u505c\u8f66", "single_vehicle_stop"

    if parking_score >= 3 and accident_score <= 1:
        return "\u5f02\u5e38\u505c\u8f66", "parking_dominant"

    if congestion_score >= 3 and accident_score <= 1:
        return "\u62e5\u5835", "congestion_dominant"

    return TARGET, ""


def main():
    ap = argparse.ArgumentParser(description="Conservative label postprocess for E8 multi-car false positives.")
    ap.add_argument("--input", required=True, help="E8 prediction jsonl")
    ap.add_argument("--output", required=True, help="Postprocessed jsonl")
    ap.add_argument("--pred-field", default="pred")
    ap.add_argument("--label-field", default="pred_post_label")
    args = ap.parse_args()

    total = 0
    changed = 0
    reasons = Counter()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.input).open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            pred = norm(rec.get(args.pred_field))
            fixed_label, reason = fix_multicar_label(pred)
            rec[args.label_field] = fixed_label
            rec["pred_postprocess_reason"] = reason
            if reason:
                changed += 1
                reasons[reason] += 1
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("input =", args.input)
    print("output =", args.output)
    print("total =", total)
    print("changed =", changed)
    print("reasons =", dict(reasons))


if __name__ == "__main__":
    main()
