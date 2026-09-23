# Frozen transplant from inference.py, holmesvau/holmesvau_utils.py, and
# holmesvau/internvl_utils.py in the third-chapter repository.
# Source commit: 8b98e6cda09d078061eb308d78feec75fa23e6fd
"""HolmesVAU loading and direct video inference without third-chapter imports."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _closest_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def dynamic_preprocess(
    image: Image.Image,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> List[Image.Image]:
    width, height = image.size
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    cols, rows = _closest_ratio(width / height, target_ratios, width, height, image_size)
    resized = image.resize((image_size * cols, image_size * rows))
    tiles = []
    for index in range(cols * rows):
        left = (index % cols) * image_size
        top = (index // cols) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def uniform_indices(frame_count: int, num_frames: int) -> List[int]:
    if frame_count <= 0:
        raise ValueError("video contains no frames")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    segment = float(frame_count - 1) / num_frames
    return [
        int(segment / 2 + round(segment * index))
        for index in range(num_frames)
    ]


def load_model(model_path: str, device: str):
    """Load HolmesVAU with eager attention; flash-attn is intentionally disabled."""
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"model_path is not a directory: {path}")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    if hasattr(config, "vision_config"):
        config.vision_config.use_flash_attn = False
    if hasattr(config, "llm_config"):
        config.llm_config.attn_implementation = "eager"
    model = AutoModel.from_pretrained(
        path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


def prepare_video(
    video_path: str,
    num_frames: int,
    input_size: int,
    max_tiles_per_frame: int,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frame_indices = uniform_indices(len(reader), num_frames)
    transform = build_transform(input_size)
    values = []
    patches = []
    for frame_index in frame_indices:
        image = Image.fromarray(reader[frame_index].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(
            image,
            min_num=1,
            max_num=max_tiles_per_frame,
            image_size=input_size,
            use_thumbnail=True,
        )
        tensor = torch.stack([transform(tile) for tile in tiles])
        values.append(tensor)
        patches.append(tensor.shape[0])
    return torch.cat(values), patches, frame_indices


@torch.inference_mode()
def generate(
    video_path: str,
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    num_frames: int,
    input_size: int,
    max_tiles_per_frame: int,
) -> Tuple[str, List[int]]:
    """Run one direct greedy query. No retry or output correction is performed."""
    if temperature != 0:
        raise ValueError("B0 requires temperature=0")
    pixel_values, num_patches_list, frame_indices = prepare_video(
        video_path, num_frames, input_size, max_tiles_per_frame
    )
    prefix = "".join(
        f"Frame{index + 1}: <image>\n" for index in range(len(num_patches_list))
    )
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    response = model.chat(
        tokenizer,
        pixel_values.to(dtype=torch.bfloat16, device=model.device),
        prefix + prompt,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )
    if isinstance(response, tuple):
        response = response[0]
    return str(response), frame_indices
