#!/usr/bin/env python3
"""
在AutoDL服务器上运行此脚本
提取正确预测案例的视频帧
"""

import cv2
import os
from pathlib import Path

# 输出目录
OUTPUT_DIR = '/root/autodl-tmp/高架桥数据/correct_predictions_frames'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 要提取的视频帧配置
FRAMES_TO_EXTRACT = [
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/拥堵_20240113182656_突发事件_35316755.mp4',
        'frame_index': 802,
        'category': '拥堵',
        'rank': 1,
        'video_name': '拥堵_20240113182656_突发事件_35316755.mp4',
        'output_name': '拥堵_rank1_拥堵_20240113182656_突发事件_35316755.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/拥堵_20240106181900_突发事件_35298453.mp4',
        'frame_index': 807,
        'category': '拥堵',
        'rank': 2,
        'video_name': '拥堵_20240106181900_突发事件_35298453.mp4',
        'output_name': '拥堵_rank2_拥堵_20240106181900_突发事件_35298453.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/拥堵_20240206163352_突发事件_35355647.mp4',
        'frame_index': 793,
        'category': '拥堵',
        'rank': 3,
        'video_name': '拥堵_20240206163352_突发事件_35355647.mp4',
        'output_name': '拥堵_rank3_拥堵_20240206163352_突发事件_35355647.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/异常停车_20240624112654_突发事件_35721711.mp4',
        'frame_index': 793,
        'category': '异常停车',
        'rank': 1,
        'video_name': '异常停车_20240624112654_突发事件_35721711.mp4',
        'output_name': '异常停车_rank1_异常停车_20240624112654_突发事件_35721711.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/异常停车_20240623150949_突发事件_35720119.mp4',
        'frame_index': 418,
        'category': '异常停车',
        'rank': 2,
        'video_name': '异常停车_20240623150949_突发事件_35720119.mp4',
        'output_name': '异常停车_rank2_异常停车_20240623150949_突发事件_35720119.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/异常停车_20240511021033_突发事件_35579528.mp4',
        'frame_index': 805,
        'category': '异常停车',
        'rank': 3,
        'video_name': '异常停车_20240511021033_突发事件_35579528.mp4',
        'output_name': '异常停车_rank3_异常停车_20240511021033_突发事件_35579528.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/二轮车辆闯入_20230721144136_突发事件_33393747.mp4',
        'frame_index': 117,
        'category': '二轮车辆闯入',
        'rank': 1,
        'video_name': '二轮车辆闯入_20230721144136_突发事件_33393747.mp4',
        'output_name': '二轮车辆闯入_rank1_二轮车辆闯入_20230721144136_突发事件_33393747.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/二轮车辆闯入_20230721120212_突发事件_33393572.mp4',
        'frame_index': 398,
        'category': '二轮车辆闯入',
        'rank': 2,
        'video_name': '二轮车辆闯入_20230721120212_突发事件_33393572.mp4',
        'output_name': '二轮车辆闯入_rank2_二轮车辆闯入_20230721120212_突发事件_33393572.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/二轮车辆闯入_20230715120335_突发事件_1149569.mp4',
        'frame_index': 392,
        'category': '二轮车辆闯入',
        'rank': 3,
        'video_name': '二轮车辆闯入_20230715120335_突发事件_1149569.mp4',
        'output_name': '二轮车辆闯入_rank3_二轮车辆闯入_20230715120335_突发事件_1149569.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/占道施工_20230820045634_运行安全_34614824.mp4',
        'frame_index': 821,
        'category': '占道施工',
        'rank': 1,
        'video_name': '占道施工_20230820045634_运行安全_34614824.mp4',
        'output_name': '占道施工_rank1_占道施工_20230820045634_运行安全_34614824.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/占道施工_20240421095734_运行安全_35504665.mp4',
        'frame_index': 413,
        'category': '占道施工',
        'rank': 2,
        'video_name': '占道施工_20240421095734_运行安全_35504665.mp4',
        'output_name': '占道施工_rank2_占道施工_20240421095734_运行安全_35504665.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/占道施工_20240430000257_运行安全_35546785.mp4',
        'frame_index': 420,
        'category': '占道施工',
        'rank': 3,
        'video_name': '占道施工_20240430000257_运行安全_35546785.mp4',
        'output_name': '占道施工_rank3_占道施工_20240430000257_运行安全_35546785.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/多车事故_20240622131800_突发事件_35718810.mp4',
        'frame_index': 172,
        'category': '多车事故',
        'rank': 1,
        'video_name': '多车事故_20240622131800_突发事件_35718810.mp4',
        'output_name': '多车事故_rank1_多车事故_20240622131800_突发事件_35718810.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/多车事故_20240530085706_突发事件_35653363.mp4',
        'frame_index': 811,
        'category': '多车事故',
        'rank': 2,
        'video_name': '多车事故_20240530085706_突发事件_35653363.mp4',
        'output_name': '多车事故_rank2_多车事故_20240530085706_突发事件_35653363.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/多车事故_20240622062856_突发事件_35717902.mp4',
        'frame_index': 800,
        'category': '多车事故',
        'rank': 3,
        'video_name': '多车事故_20240622062856_突发事件_35717902.mp4',
        'output_name': '多车事故_rank3_多车事故_20240622062856_突发事件_35717902.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/抛洒物_20240530074153_运行安全_35652730.mp4',
        'frame_index': 786,
        'category': '抛洒物',
        'rank': 1,
        'video_name': '抛洒物_20240530074153_运行安全_35652730.mp4',
        'output_name': '抛洒物_rank1_抛洒物_20240530074153_运行安全_35652730.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/抛洒物_20240216203149_运行安全_35364108.mp4',
        'frame_index': 785,
        'category': '抛洒物',
        'rank': 2,
        'video_name': '抛洒物_20240216203149_运行安全_35364108.mp4',
        'output_name': '抛洒物_rank2_抛洒物_20240216203149_运行安全_35364108.mp4'
    },
    {
        'video_path': '/root/autodl-tmp/高架桥数据/video_fixed/抛洒物_20240315101211_运行安全_35412640.mp4',
        'frame_index': 801,
        'category': '抛洒物',
        'rank': 3,
        'video_name': '抛洒物_20240315101211_运行安全_35412640.mp4',
        'output_name': '抛洒物_rank3_抛洒物_20240315101211_运行安全_35412640.mp4'
    },
]

print("="*70)
print("提取正确预测案例的视频帧")
print("="*70)

success_count = 0
fail_count = 0

for config in FRAMES_TO_EXTRACT:
    video_path = config['video_path']
    frame_index = config['frame_index']
    output_name = config['output_name']
    category = config['category']
    rank = config['rank']

    print(f"\n处理: {category} - Rank {rank}")
    print(f"  视频: {video_path}")
    print(f"  帧索引: {frame_index}")

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"  ⚠️  视频文件不存在")
        fail_count += 1
        continue

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  ⚠️  无法打开视频")
        fail_count += 1
        continue

    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"  ⚠️  无法读取帧 {frame_index}")
        fail_count += 1
        continue

    # 保存帧
    output_path = os.path.join(OUTPUT_DIR, output_name.replace('.mp4', f'_frame_{frame_index}.jpg'))
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"  ✓ 已保存: {output_path}")
    success_count += 1

print("\n" + "="*70)
print("提取完成！")
print("="*70)
print(f"成功: {success_count} 个")
print(f"失败: {fail_count} 个")
print(f"\n所有帧已保存到: {OUTPUT_DIR}")
print("\n请将此目录打包下载到本地:")
print(f"  tar -czf correct_predictions_frames.tar.gz {OUTPUT_DIR}")
