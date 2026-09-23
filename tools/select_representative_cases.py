#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从交通异常视频中自动选择各类别代表帧，保留原始 JPG，
生成带“可疑区域框 + 帧内局部放大框 + 连接线”的 JPG，
并自动拼接为 3×6 案例矩阵。

默认规则：
1. 类别取视频文件名第一个下划线 "_" 前的内容；
2. 每类处理若干不同视频；
3. 每个视频均匀抽取候选帧；
4. 综合清晰度、亮度、对比度、运动变化和显著性评分；
5. 每个视频只保留一张最佳帧；
6. 每类最终保留三张，且尽量来自不同场景；
7. 图中不显示类别名、标签、置信度、文件名或帧号；
8. 原始视频帧永久保存为 JPG。

注意：
自动框属于启发式“可疑区域”，并非模型真实检测框。
正式用于论文前应人工复核并调整少量错误框。

依赖：
    pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass
class FrameCandidate:
    category: str
    video_path: Path
    frame_index: int
    timestamp_sec: float
    fps: float
    frame_count: int
    frame: np.ndarray
    sharpness: float
    brightness_score: float
    contrast: float
    motion_score: float
    saliency_score: float
    box_score: float
    suspicious_box: Tuple[int, int, int, int]
    total_score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="选择各类别代表性视频帧并生成3×6案例矩阵"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="按给定顺序指定6个类别；不指定则自动选择视频数量最多的6类",
    )
    parser.add_argument("--videos-per-class", type=int, default=12)
    parser.add_argument("--candidate-frames-per-video", type=int, default=24)
    parser.add_argument("--final-frames-per-class", type=int, default=3)
    parser.add_argument("--preselect-per-class", type=int, default=8)
    parser.add_argument("--start-ratio", type=float, default=0.08)
    parser.add_argument("--end-ratio", type=float, default=0.92)
    parser.add_argument("--matrix-cell-width", type=int, default=480)
    parser.add_argument("--matrix-cell-height", type=int, default=270)
    parser.add_argument("--jpeg-quality", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_filename(text: str) -> str:
    for char in '<>:"/\\|?*':
        text = text.replace(char, "_")
    return text.strip().rstrip(".")


def infer_category(video_path: Path) -> str:
    stem = video_path.stem
    return stem.split("_", 1)[0].strip() if "_" in stem else video_path.parent.name.strip()


def discover_videos(dataset_root: Path) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            grouped.setdefault(infer_category(path), []).append(path)
    for category in grouped:
        grouped[category] = sorted(grouped[category])
    return grouped


def normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return arr
    low, high = float(arr.min()), float(arr.max())
    if math.isclose(low, high):
        return np.full_like(arr, 0.5)
    return (arr - low) / (high - low)


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes:02d}m{remain:05.2f}s"


def save_jpg(path: Path, image: np.ndarray, quality: int, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    ok = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))],
    )
    if not ok:
        raise RuntimeError(f"保存失败：{path}")


def brightness_score(gray: np.ndarray) -> float:
    mean = float(gray.mean())
    return max(0.0, 1.0 - abs(mean - 130.0) / 130.0)


def resize_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)


def spectral_residual_saliency(gray: np.ndarray) -> np.ndarray:
    small = resize_gray(gray).astype(np.float32)
    spectrum = np.fft.fft2(small)
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    log_amp = np.log(amplitude + 1e-8)
    residual = log_amp - cv2.blur(log_amp.astype(np.float32), (3, 3))
    reconstructed = np.exp(residual + 1j * phase)
    saliency = np.abs(np.fft.ifft2(reconstructed)) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
    saliency -= saliency.min()
    if saliency.max() > 1e-8:
        saliency /= saliency.max()
    return saliency


def motion_map(
    previous_gray: Optional[np.ndarray],
    current_gray: np.ndarray,
    next_gray: Optional[np.ndarray],
) -> np.ndarray:
    current_small = resize_gray(current_gray)
    diffs = []
    if previous_gray is not None:
        diffs.append(cv2.absdiff(current_small, resize_gray(previous_gray)))
    if next_gray is not None:
        diffs.append(cv2.absdiff(current_small, resize_gray(next_gray)))
    if not diffs:
        return np.zeros_like(current_small, dtype=np.float32)
    motion = np.mean(np.stack(diffs, axis=0), axis=0).astype(np.float32)
    motion = cv2.GaussianBlur(motion, (9, 9), 0)
    motion -= motion.min()
    if motion.max() > 1e-8:
        motion /= motion.max()
    return motion


def texture_map(gray: np.ndarray) -> np.ndarray:
    small = resize_gray(gray)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    magnitude = cv2.GaussianBlur(magnitude, (7, 7), 0)
    magnitude -= magnitude.min()
    if magnitude.max() > 1e-8:
        magnitude /= magnitude.max()
    return magnitude


def road_weight_map(height: int, width: int) -> np.ndarray:
    y = np.linspace(0.35, 1.0, height, dtype=np.float32)
    return np.repeat(y[:, None], width, axis=1)


def clamp_box(
    box: Tuple[int, int, int, int], width: int, height: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 2, int(x1)))
    y1 = max(0, min(height - 2, int(y1)))
    x2 = max(x1 + 1, min(width - 1, int(x2)))
    y2 = max(y1 + 1, min(height - 1, int(y2)))
    return x1, y1, x2, y2


def expand_box(
    box: Tuple[int, int, int, int], width: int, height: int, scale: float = 1.25
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    return clamp_box(
        (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
        width,
        height,
    )


def detect_suspicious_region(
    frame: np.ndarray,
    previous_frame: Optional[np.ndarray],
    next_frame: Optional[np.ndarray],
) -> Tuple[Tuple[int, int, int, int], float, float]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = (
        cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        if previous_frame is not None
        else None
    )
    next_gray = (
        cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        if next_frame is not None
        else None
    )

    saliency = spectral_residual_saliency(gray)
    motion = motion_map(prev_gray, gray, next_gray)
    texture = texture_map(gray)
    combined = (0.48 * saliency + 0.32 * motion + 0.20 * texture)
    combined *= road_weight_map(*combined.shape)
    combined = cv2.GaussianBlur(combined.astype(np.float32), (11, 11), 0)

    threshold = float(np.percentile(combined, 88))
    binary = (combined >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sh, sw = combined.shape
    min_area = sh * sw * 0.003
    max_area = sh * sw * 0.16
    best_box = None
    best_score = -1.0

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if not (min_area <= area <= max_area):
            continue
        region = combined[y:y + bh, x:x + bw]
        if region.size == 0:
            continue
        score = 0.6 * float(region.mean()) + 0.4 * float(region.max())
        if score > best_score:
            best_score = score
            best_box = (x, y, x + bw, y + bh)

    if best_box is None:
        py, px = np.unravel_index(np.argmax(combined), combined.shape)
        bw, bh = max(28, int(sw * 0.13)), max(24, int(sh * 0.16))
        best_box = (px - bw // 2, py - bh // 2, px + bw // 2, py + bh // 2)
        best_score = float(combined[py, px])

    sx, sy = w / sw, h / sh
    x1, y1, x2, y2 = best_box
    box = clamp_box((x1 * sx, y1 * sy, x2 * sx, y2 * sy), w, h)
    box = expand_box(box, w, h)
    return box, float(combined.max()), max(best_score, 0.0)


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def sample_video_candidates(
    category: str,
    video_path: Path,
    candidate_count: int,
    start_ratio: float,
    end_ratio: float,
) -> List[FrameCandidate]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[跳过] 无法打开：{video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or fps <= 0:
        cap.release()
        return []

    start_idx = max(0, int(frame_count * start_ratio))
    end_idx = min(frame_count - 1, int(frame_count * end_ratio))
    indices = np.unique(
        np.linspace(start_idx, end_idx, num=min(candidate_count, end_idx - start_idx + 1), dtype=int)
    )
    offset = max(1, int(round(fps * 0.5)))
    candidates: List[FrameCandidate] = []

    for idx in indices:
        current = read_frame_at(cap, int(idx))
        if current is None:
            continue
        previous = read_frame_at(cap, max(0, int(idx) - offset))
        next_frame = read_frame_at(cap, min(frame_count - 1, int(idx) + offset))
        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

        box, saliency_score, box_score = detect_suspicious_region(
            current, previous, next_frame
        )
        candidates.append(
            FrameCandidate(
                category=category,
                video_path=video_path,
                frame_index=int(idx),
                timestamp_sec=float(idx / fps),
                fps=fps,
                frame_count=frame_count,
                frame=current,
                sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                brightness_score=brightness_score(gray),
                contrast=float(gray.std()),
                motion_score=float(
                    motion_map(
                        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) if previous is not None else None,
                        gray,
                        None,
                    ).mean()
                ),
                saliency_score=saliency_score,
                box_score=box_score,
                suspicious_box=box,
            )
        )

    cap.release()
    if not candidates:
        return []

    s = normalize([c.sharpness for c in candidates])
    c = normalize([c.contrast for c in candidates])
    m = normalize([c.motion_score for c in candidates])
    a = normalize([c.saliency_score for c in candidates])
    b = normalize([c.box_score for c in candidates])

    for i, item in enumerate(candidates):
        item.total_score = float(
            0.22 * s[i]
            + 0.10 * item.brightness_score
            + 0.13 * c[i]
            + 0.15 * m[i]
            + 0.22 * a[i]
            + 0.18 * b[i]
        )
    return candidates


def perceptual_signature(frame: np.ndarray, size: int = 16) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32).reshape(-1) / 255.0


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def intersection_area(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> int:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def choose_inset_position(
    width: int,
    height: int,
    inset_width: int,
    inset_height: int,
    box: Tuple[int, int, int, int],
    margin: int,
) -> Tuple[int, int, int, int]:
    positions = [
        (margin, margin, margin + inset_width, margin + inset_height),
        (width - margin - inset_width, margin, width - margin, margin + inset_height),
        (margin, height - margin - inset_height, margin + inset_width, height - margin),
        (
            width - margin - inset_width,
            height - margin - inset_height,
            width - margin,
            height - margin,
        ),
    ]
    return min(positions, key=lambda p: intersection_area(p, box))


def draw_case_visualization(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    output = frame.copy()
    h, w = output.shape[:2]
    x1, y1, x2, y2 = box

    red = (30, 35, 220)
    yellow = (0, 235, 255)
    green = (35, 200, 60)
    thickness = max(2, int(round(min(w, h) / 260)))

    cv2.rectangle(output, (x1, y1), (x2, y2), red, thickness, cv2.LINE_AA)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return output

    inset_w = min(max(120, int(w * 0.25)), w - 24)
    inset_h = min(max(90, int(h * 0.29)), h - 24)
    margin = max(8, int(min(w, h) * 0.015))
    ix1, iy1, ix2, iy2 = choose_inset_position(
        w, h, inset_w, inset_h, box, margin
    )

    enlarged = cv2.resize(
        crop, (ix2 - ix1, iy2 - iy1), interpolation=cv2.INTER_CUBIC
    )
    output[iy1:iy2, ix1:ix2] = enlarged
    cv2.rectangle(output, (ix1, iy1), (ix2, iy2), yellow, thickness, cv2.LINE_AA)

    mx = max(5, int((ix2 - ix1) * 0.08))
    my = max(5, int((iy2 - iy1) * 0.08))
    cv2.rectangle(
        output,
        (ix1 + mx, iy1 + my),
        (ix2 - mx, iy2 - my),
        green,
        thickness,
        cv2.LINE_AA,
    )

    source = ((x1 + x2) // 2, (y1 + y2) // 2)
    target = (
        ix1 if source[0] < (ix1 + ix2) // 2 else ix2,
        iy1 if source[1] < (iy1 + iy2) // 2 else iy2,
    )
    cv2.line(output, source, target, yellow, max(1, thickness - 1), cv2.LINE_AA)
    return output


def resize_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(
        image,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    rh, rw = resized.shape[:2]
    x = max(0, (rw - width) // 2)
    y = max(0, (rh - height) // 2)
    return resized[y:y + height, x:x + width]


def make_candidate_sheet(
    candidates: List[FrameCandidate],
    output_path: Path,
) -> None:
    if not candidates:
        return
    cells = [
        resize_crop(draw_case_visualization(c.frame, c.suspicious_box), 420, 236)
        for c in candidates
    ]
    columns = 4
    rows = math.ceil(len(cells) / columns)
    blank = np.full((236, 420, 3), 245, dtype=np.uint8)
    while len(cells) < rows * columns:
        cells.append(blank.copy())
    sheet = np.vstack(
        [np.hstack(cells[r * columns:(r + 1) * columns]) for r in range(rows)]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])


def build_matrix(
    selected: Dict[str, List[FrameCandidate]],
    class_order: List[str],
    output_root: Path,
    cell_width: int,
    cell_height: int,
) -> None:
    rows = 3
    gap = max(3, int(round(cell_width * 0.008)))
    h_gap = np.full((cell_height, gap, 3), 255, dtype=np.uint8)
    total_width = len(class_order) * cell_width + (len(class_order) - 1) * gap
    v_gap = np.full((gap, total_width, 3), 255, dtype=np.uint8)
    matrix_rows = []

    for row_idx in range(rows):
        row_cells = []
        for col_idx, category in enumerate(class_order):
            items = selected.get(category, [])
            if row_idx < len(items):
                cell = resize_crop(
                    draw_case_visualization(
                        items[row_idx].frame,
                        items[row_idx].suspicious_box,
                    ),
                    cell_width,
                    cell_height,
                )
            else:
                cell = np.full((cell_height, cell_width, 3), 235, dtype=np.uint8)
            row_cells.append(cell)
            if col_idx < len(class_order) - 1:
                row_cells.append(h_gap.copy())
        matrix_rows.append(np.hstack(row_cells))
        if row_idx < rows - 1:
            matrix_rows.append(v_gap.copy())

    matrix = np.vstack(matrix_rows)
    cv2.imwrite(
        str(output_root / "representative_cases_3x6.jpg"),
        matrix,
        [cv2.IMWRITE_JPEG_QUALITY, 97],
    )
    cv2.imwrite(
        str(output_root / "representative_cases_3x6.png"),
        matrix,
        [cv2.IMWRITE_PNG_COMPRESSION, 2],
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if not (0 <= args.start_ratio < args.end_ratio <= 1):
        raise ValueError("必须满足 0 <= start-ratio < end-ratio <= 1")

    grouped = discover_videos(dataset_root)
    if not grouped:
        raise RuntimeError("未找到视频文件")

    print("发现类别：")
    for category, videos in grouped.items():
        print(f"  {category}: {len(videos)}")

    if args.classes:
        class_order = args.classes
        missing = [c for c in class_order if c not in grouped]
        if missing:
            raise ValueError(f"未找到以下类别：{missing}")
    else:
        class_order = [
            c for c, _ in sorted(
                grouped.items(), key=lambda item: len(item[1]), reverse=True
            )[:6]
        ]

    output_root.mkdir(parents=True, exist_ok=True)
    selected_by_class: Dict[str, List[FrameCandidate]] = {}
    records: List[dict] = []

    for class_idx, category in enumerate(class_order, start=1):
        videos = grouped[category].copy()
        random.shuffle(videos)
        videos = videos[:args.videos_per_class]

        print(f"\n[{class_idx}/{len(class_order)}] {category}")
        best_per_video: List[FrameCandidate] = []

        for video_idx, video_path in enumerate(videos, start=1):
            print(f"  [{video_idx}/{len(videos)}] {video_path.name}")
            candidates = sample_video_candidates(
                category,
                video_path,
                args.candidate_frames_per_video,
                args.start_ratio,
                args.end_ratio,
            )
            if candidates:
                best_per_video.append(max(candidates, key=lambda x: x.total_score))

        best_per_video.sort(key=lambda x: x.total_score, reverse=True)
        preselected = best_per_video[:args.preselect_per_class]

        make_candidate_sheet(
            preselected,
            output_root
            / "candidate_sheets"
            / f"{safe_filename(category)}_candidates.jpg",
        )

        final: List[FrameCandidate] = []
        signatures: List[np.ndarray] = []

        for candidate in preselected:
            sig = perceptual_signature(candidate.frame)
            if any(signature_distance(sig, old) < 0.035 for old in signatures):
                continue
            final.append(candidate)
            signatures.append(sig)
            if len(final) >= args.final_frames_per_class:
                break

        if len(final) < args.final_frames_per_class:
            used = {item.video_path for item in final}
            for candidate in preselected:
                if candidate.video_path in used:
                    continue
                final.append(candidate)
                used.add(candidate.video_path)
                if len(final) >= args.final_frames_per_class:
                    break

        selected_by_class[category] = final

        for rank, candidate in enumerate(final, start=1):
            base = (
                f"{rank:02d}_"
                f"{safe_filename(candidate.video_path.stem)}_"
                f"{format_timestamp(candidate.timestamp_sec)}_"
                f"frame_{candidate.frame_index:08d}"
            )

            clean_path = (
                output_root
                / "clean_frames"
                / safe_filename(category)
                / f"{base}_clean.jpg"
            )
            annotated_path = (
                output_root
                / "annotated_frames"
                / safe_filename(category)
                / f"{base}_annotated.jpg"
            )

            save_jpg(
                clean_path,
                candidate.frame,
                args.jpeg_quality,
                args.overwrite,
            )
            save_jpg(
                annotated_path,
                draw_case_visualization(
                    candidate.frame,
                    candidate.suspicious_box,
                ),
                args.jpeg_quality,
                args.overwrite,
            )

            x1, y1, x2, y2 = candidate.suspicious_box
            records.append(
                {
                    "category": category,
                    "rank": rank,
                    "video_path": str(candidate.video_path),
                    "frame_index": candidate.frame_index,
                    "timestamp_sec": round(candidate.timestamp_sec, 4),
                    "total_score": round(candidate.total_score, 6),
                    "box_x1": x1,
                    "box_y1": y1,
                    "box_x2": x2,
                    "box_y2": y2,
                    "clean_frame_path": str(clean_path),
                    "annotated_frame_path": str(annotated_path),
                }
            )

    if records:
        csv_path = output_root / "selected_frames.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    build_matrix(
        selected_by_class,
        class_order,
        output_root,
        args.matrix_cell_width,
        args.matrix_cell_height,
    )

    print("\n完成。")
    print(f"原始帧：{output_root / 'clean_frames'}")
    print(f"标注帧：{output_root / 'annotated_frames'}")
    print(f"候选总览：{output_root / 'candidate_sheets'}")
    print(f"矩阵图：{output_root / 'representative_cases_3x6.png'}")


if __name__ == "__main__":
    main()
