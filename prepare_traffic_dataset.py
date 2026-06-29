import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd


def normalize_path(path: str) -> str:
    return path.replace('\\', '/').strip()


def resolve_existing_path(path: str) -> str:
    """Resolve both absolute paths and paths relative to current dir or root (/)."""
    raw = Path(path).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)
        candidates.append(Path('/') / raw)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    # Fall back to cwd-relative absolute path for clearer downstream errors.
    return str((Path.cwd() / raw).resolve())


def to_relative(video_path: str, root: str) -> str:
    video_path = normalize_path(video_path)
    root = normalize_path(root)
    if os.path.isabs(video_path):
        try:
            rel = Path(video_path).resolve().relative_to(Path(root).resolve())
            return str(rel).replace('\\', '/')
        except Exception:
            return video_path
    return video_path


def extract_category(video_name: str) -> str:
    if not video_name:
        return ''
    basename = os.path.basename(video_name)
    if '_' in basename:
        return basename.split('_', 1)[0]
    return basename


def build_prompt(category: str, lighting: str, location: str, user_prompt: str = None) -> str:
    if user_prompt and str(user_prompt).strip():
        return str(user_prompt).strip()

    pieces = []
    if category:
        pieces.append(f'事故类型是“{category}”')
    if location:
        pieces.append(f'发生地点是{location}')
    if lighting:
        pieces.append(f'光照条件是{lighting}')

    if pieces:
        context = '，'.join(pieces)
        return f'<video>\n请根据视频内容解释该交通事故的发生过程、原因及影响。{context}。'
    return '<video>\n请根据视频内容解释该交通事故的发生过程、原因及影响。'


def build_record(idx: int, video_rel: str, prompt: str, answer: str, task: str = 'analysis') -> dict:
    return {
        'id': idx,
        'type': 'video',
        'task': task,
        'video': video_rel,
        'conversations': [
            {'from': 'human', 'value': prompt},
            {'from': 'gpt', 'value': answer}
        ]
    }


def save_jsonl(records, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def find_first_existing_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f'None of columns {candidates} found. Existing columns: {df.columns.tolist()}')
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert traffic annotation Excel to HolmesVAU jsonl/meta (AutoDL friendly).'
    )
    parser.add_argument('--excel', default='autodl-tmp/高架桥数据/标注.xlsx', help='Path to annotation Excel file')
    parser.add_argument('--video-root', default='autodl-tmp/高架桥数据/video', help='Root directory containing videos')
    parser.add_argument('--output-dir', default='autodl-tmp/高架桥数据', help='Output directory for train/test jsonl and meta json')

    parser.add_argument('--video-column', default=None, help='Video column name (auto-detect if omitted)')
    parser.add_argument('--text-column', default=None, help='Annotation text column name (auto-detect if omitted)')
    parser.add_argument('--prompt-column', default=None, help='Optional custom prompt column name')
    parser.add_argument('--lighting-column', default=None, help='Optional lighting column name')
    parser.add_argument('--location-column', default=None, help='Optional location column name')
    parser.add_argument('--split-column', default=None, help='Optional split column containing train/test labels')

    parser.add_argument('--task', default='analysis', help='Task field in output records')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio when split column is not provided')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for random splitting')
    parser.add_argument('--meta-name', default='traffic_meta.json', help='Output meta file name')
    return parser.parse_args()


def main():
    args = parse_args()

    excel_path = resolve_existing_path(args.excel)
    video_root = resolve_existing_path(args.video_root)
    output_dir = Path(resolve_existing_path(args.output_dir)) if Path(args.output_dir).exists() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(excel_path)

    video_col = args.video_column or find_first_existing_column(
        df, ['视频名称', '视频名', 'video', 'video_name', 'filename']
    )
    text_col = args.text_column or find_first_existing_column(
        df, ['修改后标注结果', '标注结果', '文本', 'text', 'answer', 'annotation']
    )

    prompt_col = args.prompt_column if args.prompt_column in df.columns else find_first_existing_column(
        df, ['prompt', 'question', '问题'], required=False
    )
    lighting_col = args.lighting_column if args.lighting_column in df.columns else find_first_existing_column(
        df, ['光照', 'lighting'], required=False
    )
    location_col = args.location_column if args.location_column in df.columns else find_first_existing_column(
        df, ['地点', 'location'], required=False
    )
    split_col = args.split_column if args.split_column in df.columns else find_first_existing_column(
        df, ['split', 'set', '数据集划分', '训练测试划分'], required=False
    )

    df = df.dropna(subset=[video_col, text_col])

    rows = []
    for _, row in df.iterrows():
        video_path = str(row[video_col]).strip()
        annotation = str(row[text_col]).strip()
        prompt_text = str(row[prompt_col]).strip() if prompt_col else None
        lighting = str(row[lighting_col]).strip() if lighting_col else ''
        location = str(row[location_col]).strip() if location_col else ''
        category = extract_category(video_path)

        prompt = build_prompt(category, lighting, location, prompt_text)
        video_rel = to_relative(video_path, video_root)
        rows.append((video_rel, prompt, annotation, row))

    if split_col:
        train_records = []
        test_records = []
        for idx, (video_rel, prompt, annotation, row) in enumerate(rows):
            label = str(row[split_col]).strip().lower()
            if label in {'train', 'trainset', 'tr', '训练', '训练集'}:
                train_records.append(build_record(idx, video_rel, prompt, annotation, task=args.task))
            elif label in {'test', 'testset', 'te', '验证', '测试', '测试集', 'val', 'valid'}:
                test_records.append(build_record(idx, video_rel, prompt, annotation, task=args.task))
            else:
                raise ValueError(f'Unknown split label "{label}" in split column "{split_col}"')
    else:
        random.seed(args.seed)
        random.shuffle(rows)
        split_index = int(len(rows) * args.train_ratio)
        train_records = [
            build_record(i, rows[i][0], rows[i][1], rows[i][2], task=args.task)
            for i in range(split_index)
        ]
        test_records = [
            build_record(split_index + i, rows[split_index + i][0], rows[split_index + i][1], rows[split_index + i][2],
                         task=args.task)
            for i in range(len(rows) - split_index)
        ]

    train_path = output_dir / 'traffic_train.jsonl'
    test_path = output_dir / 'traffic_test.jsonl'
    meta_path = output_dir / args.meta_name

    save_jsonl(train_records, str(train_path))
    save_jsonl(test_records, str(test_path))

    meta = {
        'TrafficAccident': {
            'root': normalize_path(str(Path(video_root).resolve())),
            'annotation': normalize_path(str(train_path.resolve())),
            'data_augment': False,
            'repeat_time': 1,
            'length': len(train_records)
        }
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('Saved:')
    print('  excel        :', excel_path)
    print('  video_root   :', Path(video_root).resolve())
    print('  train_jsonl  :', train_path.resolve())
    print('  test_jsonl   :', test_path.resolve())
    print('  meta_json    :', meta_path.resolve())
    print('  train samples:', len(train_records))
    print('  test samples :', len(test_records))


if __name__ == '__main__':
    main()
