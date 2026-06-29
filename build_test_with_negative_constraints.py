import argparse
import json
from pathlib import Path


NEGATIVE_CONSTRAINT = """[分类边界约束]
请特别区分“多车事故”和其他相似异常：
1. 单车静止、单车占道、单车停在应急车道或行车道时，优先判断为“异常停车”；不要仅因为后方车辆绕行、减速或局部受阻就判断为“多车事故”。
2. 大范围车辆整体低速、排队、密集缓行或多车同时停滞时，优先判断为“拥堵”；不要仅因为画面中车辆很多或车流缓慢就判断为“多车事故”。
3. 出现摩托车、电动车、非机动车进入高速或主路时，优先判断为“二轮车辆闯入”；除非明确看到两辆及以上机动车发生碰撞、追尾或事故后共同滞留，否则不要判断为“多车事故”。
4. 只有明确存在两辆及以上机动车碰撞、追尾、事故后共同停留、事故现场处置等证据时，才判断为“多车事故”。
5. 如果证据不足，请选择更具体且更保守的类别，不要臆测碰撞。"""


def inject_text(old: str, constraint: str) -> str:
    old = "" if old is None else str(old)
    if constraint.splitlines()[0] in old:
        return old
    return constraint.rstrip() + "\n" + old.lstrip()


def update_sample(sample: dict, constraint: str) -> bool:
    changed = False

    if sample.get("prompt"):
        new_prompt = inject_text(sample.get("prompt", ""), constraint)
        changed = changed or new_prompt != sample.get("prompt")
        sample["prompt"] = new_prompt

    convs = sample.get("conversations")
    if isinstance(convs, list):
        for turn in convs:
            if not isinstance(turn, dict):
                continue
            if turn.get("from") == "human":
                old = turn.get("value", "")
                new = inject_text(old, constraint)
                changed = changed or new != old
                turn["value"] = new
                break

    sample["negative_constraint_injected"] = True
    return changed


def main():
    ap = argparse.ArgumentParser(
        description="Build an inference-only test JSONL with negative class-boundary constraints."
    )
    ap.add_argument("--input", required=True, help="Input test jsonl, e.g. traffic_test_with_ovd_prompt_v8.jsonl")
    ap.add_argument("--output", required=True, help="Output test jsonl for E9 inference")
    ap.add_argument(
        "--constraint",
        default="",
        help="Optional custom constraint text. If omitted, use the built-in multi-car negative constraint.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    constraint = args.constraint if args.constraint.strip() else NEGATIVE_CONSTRAINT

    total = 0
    changed = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            if update_sample(sample, constraint):
                changed += 1
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")
            total += 1

    print("input =", in_path)
    print("output =", out_path)
    print("total =", total)
    print("changed =", changed)
    print("constraint_head =", constraint.splitlines()[0])


if __name__ == "__main__":
    main()
