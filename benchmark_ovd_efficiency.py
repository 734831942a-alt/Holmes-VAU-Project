import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import torch

import ovd_video_summary as ovd


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values, q):
    if not values:
        return None

    values = sorted(values)
    pos = (len(values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return float(values[lower])

    weight = pos - lower
    return float(
        values[lower] * (1.0 - weight)
        + values[upper] * weight
    )


def summarize_metric(values):
    values = [
        float(x) for x in values
        if x is not None
    ]

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 0.95),
        "std": (
            float(statistics.pstdev(values))
            if len(values) > 1 else 0.0
        ),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def detect_frame_timed(
    frame,
    processor,
    model,
    device,
    box_threshold,
    text_threshold,
    queries=None,
):
    queries = list(queries or ovd.TEXT_QUERIES)

    timing = {
        "preprocess_ms": 0.0,
        "forward_ms": 0.0,
        "postprocess_ms": 0.0,
        "frame_logic_ms": 0.0,
    }

    # 1. 图像转换、tokenization和数据传输
    cuda_sync()
    t0 = time.perf_counter()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    text_input = ". ".join(queries)

    inputs = processor(
        images=rgb,
        text=text_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    cuda_sync()
    timing["preprocess_ms"] = (
        time.perf_counter() - t0
    ) * 1000.0

    # 2. Grounding DINO模型前向传播
    cuda_sync()
    t1 = time.perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs)

    cuda_sync()
    timing["forward_ms"] = (
        time.perf_counter() - t1
    ) * 1000.0

    # 3. Grounding DINO后处理
    target_sizes = torch.tensor(
        [rgb.shape[:2]],
        device=device,
    )

    cuda_sync()
    t2 = time.perf_counter()

    try:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            target_sizes=target_sizes,
        )[0]

    cuda_sync()
    timing["postprocess_ms"] = (
        time.perf_counter() - t2
    ) * 1000.0

    # 4. 检测框分类、统计和去重
    t3 = time.perf_counter()

    label_list = (
        results.get("text_labels")
        or [
            str(label)
            for label in results.get("labels", [])
        ]
    )
    boxes = results.get("boxes", [])

    frame_cnt = {
        query: 0 for query in queries
    }
    frame_vehicle_boxes = []
    frame_context_boxes = {
        query: []
        for query in ovd.VSCM_CONTEXT_QUERIES
    }
    frame_boxes_by_label = {
        query: []
        for query in queries
    }

    for idx, label in enumerate(label_list):
        label = str(label)
        box = None

        if idx < len(boxes):
            raw_box = boxes[idx]
            box = [
                float(raw_box[0]),
                float(raw_box[1]),
                float(raw_box[2]),
                float(raw_box[3]),
            ]

        if label in frame_cnt:
            frame_cnt[label] += 1

        if (
            label in frame_boxes_by_label
            and box is not None
        ):
            frame_boxes_by_label[label].append(box)

        if (
            label in ovd.VEHICLE_QUERIES
            and box is not None
        ):
            frame_vehicle_boxes.append(box)

        if (
            label in ovd.VSCM_CONTEXT_QUERIES
            and box is not None
        ):
            frame_context_boxes[label].append(box)

    frame_vehicle_boxes = ovd._dedupe_vehicle_boxes(
        frame_vehicle_boxes
    )

    timing["frame_logic_ms"] = (
        time.perf_counter() - t3
    ) * 1000.0

    return (
        frame_cnt,
        frame_vehicle_boxes,
        frame_context_boxes,
        frame_boxes_by_label,
        timing,
    )


def summarize_video_timed(
    video_path,
    processor,
    model,
    device,
    box_threshold=0.30,
    text_threshold=0.25,
    frames=12,
):
    cuda_sync()

    if torch.cuda.is_available():
        baseline_allocated = (
            torch.cuda.memory_allocated()
        )
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_allocated = 0

    total_start = time.perf_counter()

    # 打开视频和读取元数据
    open_start = time.perf_counter()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    num_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    img_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    img_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    open_ms = (
        time.perf_counter() - open_start
    ) * 1000.0

    # 计算采样位置
    index_start = time.perf_counter()
    sampled_indices = set(
        ovd.sample_indices(
            num_video_frames,
            frames,
        )
    )
    index_ms = (
        time.perf_counter() - index_start
    ) * 1000.0

    labels_all = {
        query: 0
        for query in ovd.TEXT_QUERIES
    }
    labels_peak = {
        query: 0
        for query in ovd.TEXT_QUERIES
    }
    labels_first_frame = {
        query: None
        for query in ovd.TEXT_QUERIES
    }
    labels_last_frame = {
        query: None
        for query in ovd.TEXT_QUERIES
    }

    per_frame_vehicle_boxes = []
    per_frame_context_boxes = []
    per_frame_boxes_by_label = []
    per_frame_sarp_boxes_by_label = []

    used = 0
    frame_index = 0
    bad_reads = 0
    sample_sequence = 0

    decode_ms = 0.0
    preprocess_ms = 0.0
    forward_ms = 0.0
    postprocess_ms = 0.0
    frame_logic_ms = 0.0

    while True:
        decode_start = time.perf_counter()
        ok, frame = cap.read()
        decode_ms += (
            time.perf_counter() - decode_start
        ) * 1000.0

        if not ok:
            bad_reads += 1
            if bad_reads > 5:
                break

            frame_index += 1
            continue

        bad_reads = 0

        if frame_index in sampled_indices:
            (
                frame_cnt,
                frame_vehicle_boxes,
                frame_context_boxes,
                frame_boxes_by_label,
                frame_timing,
            ) = detect_frame_timed(
                frame=frame,
                processor=processor,
                model=model,
                device=device,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                queries=ovd.TEXT_QUERIES,
            )

            preprocess_ms += (
                frame_timing["preprocess_ms"]
            )
            forward_ms += (
                frame_timing["forward_ms"]
            )
            postprocess_ms += (
                frame_timing["postprocess_ms"]
            )
            frame_logic_ms += (
                frame_timing["frame_logic_ms"]
            )

            used += 1
            per_frame_vehicle_boxes.append(
                frame_vehicle_boxes
            )
            per_frame_context_boxes.append(
                frame_context_boxes
            )
            per_frame_boxes_by_label.append(
                frame_boxes_by_label
            )

            # 默认不启用SARP增强，
            # 但仍保持原脚本的普通SARP聚合输入
            per_frame_sarp_boxes_by_label.append(
                frame_boxes_by_label
            )

            for key, value in frame_cnt.items():
                labels_all[key] += value

                if value > labels_peak[key]:
                    labels_peak[key] = value

                if value > 0:
                    if labels_first_frame[key] is None:
                        labels_first_frame[key] = (
                            sample_sequence
                        )

                    labels_last_frame[key] = (
                        sample_sequence
                    )

            sample_sequence += 1

        frame_index += 1

    cap.release()

    # 跨帧证据聚合
    aggregation_start = time.perf_counter()

    lcrm = ovd._compute_lcrm_v1(
        per_frame_vehicle_boxes,
        img_width,
        img_height,
        labels_peak,
        labels_all,
    )

    vscm_result = ovd._compute_vscm_v3(
        per_frame_vehicle_boxes,
        per_frame_context_boxes,
        img_width,
        img_height,
    )

    sarp = ovd._compute_sarp(
        per_frame_sarp_boxes_by_label,
        img_width,
        img_height,
    )
    sarp["enhanced"] = False
    sarp["sampled_frames"] = len(
        per_frame_sarp_boxes_by_label
    )

    vscm_debug = None

    if (
        vscm_result
        and vscm_result.get("debug_only")
    ):
        vscm_debug = vscm_result
        vscm = None
    else:
        vscm = vscm_result

    if vscm and vscm.get("triggered", False):
        motorcycle_peak = labels_peak.get(
            "motorcycle",
            0,
        )
        motorcycle_sum = labels_all.get(
            "motorcycle",
            0,
        )

        vscm["motorcycle_peak"] = motorcycle_peak
        vscm["motorcycle_sum"] = motorcycle_sum

        if motorcycle_peak >= 1:
            vscm[
                "triggered_before_motorcycle_suppression"
            ] = True
            vscm[
                "suppression_reason"
            ] = "motorcycle_evidence"
            vscm["triggered"] = False

    aggregation_ms = (
        time.perf_counter() - aggregation_start
    ) * 1000.0

    cuda_sync()
    total_ms = (
        time.perf_counter() - total_start
    ) * 1000.0

    if torch.cuda.is_available():
        peak_allocated = (
            torch.cuda.max_memory_allocated()
        )
        peak_allocated_gb = (
            peak_allocated / 1024**3
        )
        incremental_peak_gb = max(
            0,
            peak_allocated - baseline_allocated,
        ) / 1024**3
    else:
        peak_allocated_gb = 0.0
        incremental_peak_gb = 0.0

    result = {
        "sum_count": labels_all,
        "peak_count": labels_peak,
        "first_frame": {
            key: value
            for key, value
            in labels_first_frame.items()
            if value is not None
        },
        "last_frame": {
            key: value
            for key, value
            in labels_last_frame.items()
            if value is not None
        },
        "sampled_frames": used,
        "lcrm": lcrm,
        "vscm": vscm,
        "vscm_debug": vscm_debug,
        "sarp": sarp,
    }

    detector_pipeline_ms = (
        preprocess_ms
        + forward_ms
        + postprocess_ms
        + frame_logic_ms
    )

    timing = {
        "open_ms": open_ms,
        "index_ms": index_ms,
        "decode_ms": decode_ms,
        "preprocess_ms": preprocess_ms,
        "forward_ms": forward_ms,
        "postprocess_ms": postprocess_ms,
        "frame_logic_ms": frame_logic_ms,
        "aggregation_ms": aggregation_ms,
        "detector_pipeline_ms": detector_pipeline_ms,
        "ovd_total_ms": total_ms,
        "peak_allocated_gb": peak_allocated_gb,
        "incremental_peak_gb": incremental_peak_gb,
        "total_video_frames": num_video_frames,
        "sampled_frames": used,
    }

    return result, timing


def load_rows(jsonl_path):
    rows = []

    with open(
        jsonl_path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def run_one(
    obj,
    video_root,
    processor,
    model,
    device,
    frames,
    box_threshold,
    text_threshold,
):
    video = obj.get("video", "")
    video_path = video_root / video

    record = {
        "video": video,
        "ok": False,
        "error": "",
    }

    if not video_path.exists():
        record["error"] = "video_not_found"
        return record

    try:
        _, timing = summarize_video_timed(
            video_path=video_path,
            processor=processor,
            model=model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            frames=frames,
        )

        record.update(timing)
        record["ok"] = True

    except Exception as exc:
        record["error"] = str(exc)

    return record


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl",
        required=True,
    )
    parser.add_argument(
        "--video-root",
        required=True,
    )
    parser.add_argument(
        "--out-csv",
        required=True,
    )
    parser.add_argument(
        "--out-summary",
        required=True,
    )
    parser.add_argument(
        "--model-id",
        default=(
            "IDEA-Research/"
            "grounding-dino-tiny"
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--box-thres",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--text-thres",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "0 means benchmark every record "
            "in the JSONL file."
        ),
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Model:", args.model_id)

    load_start = time.perf_counter()

    processor = ovd._Proc.from_pretrained(
        args.model_id,
        local_files_only=True,
    )
    model = ovd._Model.from_pretrained(
        args.model_id,
        local_files_only=True,
    ).to(device).eval()

    cuda_sync()

    model_load_seconds = (
        time.perf_counter() - load_start
    )

    print(
        f"Model load time: "
        f"{model_load_seconds:.2f} s"
    )

    rows = load_rows(args.jsonl)
    video_root = Path(args.video_root)

    if args.max_samples > 0:
        benchmark_rows = rows[
            :args.max_samples
        ]
    else:
        benchmark_rows = rows

    warmup_count = min(
        args.warmup_samples,
        len(rows),
    )

    print(
        f"Warm-up samples: {warmup_count}"
    )

    for index in range(warmup_count):
        video = rows[index].get("video", "")
        print(
            f"Warm-up "
            f"{index + 1}/{warmup_count}: "
            f"{video}"
        )

        run_one(
            obj=rows[index],
            video_root=video_root,
            processor=processor,
            model=model,
            device=device,
            frames=args.frames,
            box_threshold=args.box_thres,
            text_threshold=args.text_thres,
        )

    print(
        f"Benchmark samples: "
        f"{len(benchmark_rows)}"
    )

    records = []

    for index, obj in enumerate(
        benchmark_rows,
        start=1,
    ):
        record = run_one(
            obj=obj,
            video_root=video_root,
            processor=processor,
            model=model,
            device=device,
            frames=args.frames,
            box_threshold=args.box_thres,
            text_threshold=args.text_thres,
        )

        records.append(record)

        if record.get("ok"):
            print(
                f"[{index}/{len(benchmark_rows)}] "
                f"OK "
                f"{record['video']} | "
                f"total="
                f"{record['ovd_total_ms']:.2f} ms | "
                f"forward="
                f"{record['forward_ms']:.2f} ms"
            )
        else:
            print(
                f"[{index}/{len(benchmark_rows)}] "
                f"FAILED "
                f"{record['video']} | "
                f"{record.get('error', '')}"
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "video",
        "ok",
        "error",
        "total_video_frames",
        "sampled_frames",
        "open_ms",
        "index_ms",
        "decode_ms",
        "preprocess_ms",
        "forward_ms",
        "postprocess_ms",
        "frame_logic_ms",
        "aggregation_ms",
        "detector_pipeline_ms",
        "ovd_total_ms",
        "peak_allocated_gb",
        "incremental_peak_gb",
    ]

    with out_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record in records:
            writer.writerow({
                field: record.get(field, "")
                for field in fieldnames
            })

    valid_records = [
        record
        for record in records
        if record.get("ok")
    ]

    metric_names = [
        "open_ms",
        "index_ms",
        "decode_ms",
        "preprocess_ms",
        "forward_ms",
        "postprocess_ms",
        "frame_logic_ms",
        "aggregation_ms",
        "detector_pipeline_ms",
        "ovd_total_ms",
        "peak_allocated_gb",
        "incremental_peak_gb",
    ]

    summary = {
        "model_id": args.model_id,
        "device": device,
        "model_load_seconds": (
            model_load_seconds
        ),
        "frames_per_video": args.frames,
        "box_threshold": args.box_thres,
        "text_threshold": args.text_thres,
        "warmup_samples": warmup_count,
        "requested_samples": len(
            benchmark_rows
        ),
        "successful_samples": len(
            valid_records
        ),
        "failed_samples": (
            len(records)
            - len(valid_records)
        ),
        "metrics": {},
    }

    for metric in metric_names:
        summary["metrics"][metric] = (
            summarize_metric([
                record.get(metric)
                for record in valid_records
            ])
        )

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_summary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved CSV:", out_csv)
    print("Saved summary:", out_summary)
    print(
        "Successful samples:",
        len(valid_records),
    )
    print(
        "Failed samples:",
        len(records) - len(valid_records),
    )

    if valid_records:
        print("\nKey results:")

        for metric in [
            "forward_ms",
            "detector_pipeline_ms",
            "decode_ms",
            "aggregation_ms",
            "ovd_total_ms",
            "peak_allocated_gb",
        ]:
            result = summary["metrics"][metric]

            print(
                f"{metric}: "
                f"mean={result['mean']:.2f}, "
                f"median={result['median']:.2f}, "
                f"p95={result['p95']:.2f}"
            )


if __name__ == "__main__":
    main()
