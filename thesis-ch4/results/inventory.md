# 服务器与第三章资产盘点

- 盘点时间：2026-09-14（Asia/Shanghai）
- 目标工作区：`/root/thesis-ch4`
- 操作范围：仅只读盘点；除本文件 `/root/thesis-ch4/results/inventory.md` 外未写入、修改或安装任何内容。

## 1. Conda、Python 与 GPU 环境

项目 Conda 环境：`holmesvau`（路径 `/root/miniconda3/envs/holmesvau`）。

说明：非交互 SSH 登录 shell 未自动激活 Conda（`CONDA_DEFAULT_ENV` 为空，裸 `python` 不在 PATH）；下列版本均用项目环境解释器实测。

| 项目 | 版本 / 状态 |
|---|---|
| Python | 3.10.20 |
| torch | 2.11.0+cu128 |
| transformers | 4.46.3 |
| peft | 0.7.1 |
| accelerate | 0.27.2 |
| flash-attn | 未安装（`pip show` 无结果；导入报 `ModuleNotFoundError`） |
| `torch.cuda.is_available()` | `True` |
| CUDA 设备数 | 1 |
| GPU | NVIDIA GeForce RTX 5090（32607 MiB） |

## 2. 第三章（TRB）代码目录

- 路径：`/root/autodl-tmp/HolmesVAU-master`
- Git commit：`8b98e6cda09d078061eb308d78feec75fa23e6fd`

### 两层目录结构

```text
/root/autodl-tmp/HolmesVAU-master/
├── .git/
├── .vscode/
├── HIVAU-70k/
│   ├── instruction/
│   ├── raw_annotations/
│   └── videos/
├── assets/
├── autodl-tmp/
│   └── 高架桥数据/
├── bridge_experiment_results/
│   ├── by_class_train*/
│   ├── e8_multicar_case_study/
│   └── e9_multicar_clean_v1/
├── by_class_train/
├── ckpts/
│   ├── HolmesVAU-2B/
│   ├── InternVL2-2B/
│   └── HolmesVAU_lora*/
├── examples/
│   └── HIVAU-70k/
├── experiment_records/
│   ├── HolmesVAU_lora*/
│   ├── merged_evaluations/
│   └── root_logs/
├── holmesvau/
│   └── ATS/
├── internvl_chat/
│   ├── internvl/
│   └── shell/
└── tools/
```

根目录另有 `inference.py`、`ovd_video_summary.py`、`eval_traffic_fixed.py`、`eval_traffic_abnormal_only.py`、`prepare_traffic_dataset.py` 及多组 `build_*`、`postprocess_*`、`audit_*`、`simulate_*` 脚本。

### 关键入口与定义

- HolmesVAU-2B 加载与推理入口：`/root/autodl-tmp/HolmesVAU-master/inference.py`；实际加载/生成实现为 `holmesvau/holmesvau_utils.py`。
- LoRA 训练脚本入口：`internvl_chat/shell/internvl2_2b_finetune_lora.sh`；其通过 `torchrun` 调用 `internvl_chat/internvl/train/internvl_chat_finetune.py`。
- Grounding DINO 调用封装：`ovd_video_summary.py`，默认模型 `IDEA-Research/grounding-dino-tiny`。
- 开放词汇检测词表：`ovd_video_summary.py` 顶部的 `TEXT_QUERIES`，并有 `VSCM_CONTEXT_QUERIES`、`SARP_TARGET_QUERIES` 等扩展词表。
- 六类异常标签定义：`eval_traffic_fixed.py` 的 `LABELS` 与 `ALIASES`，六类为多车事故、拥堵、异常停车、占道施工、二轮车辆闯入、抛洒物。
- 六类与 OVD 信号的语义映射/提示构造：`build_ovd_prompt_zh.py`。
- 主评测脚本：`eval_traffic_fixed.py`（协议 `traffic_abnormal_fixed_v1`）；另有旧/专项评测 `eval_traffic_abnormal_only.py`、`eval_traffic_detection.py`。

因此，“六类异常的开放词汇词表”分布在三处：底层英文查询在 `ovd_video_summary.py`，六类集合及别名在 `eval_traffic_fixed.py`，语义对应在 `build_ovd_prompt_zh.py`。

## 3. HolmesVAU-2B 权重

- 实际目录：`/root/autodl-tmp/HolmesVAU-master/ckpts/HolmesVAU-2B`
- 总大小：`4,430,158,263` bytes（约 `4.13 GiB`；`du -sh` 为 `4.2G`）
- 主权重 `model.safetensors`：`4,411,571,040` bytes（约 `4.11 GiB`）
- 另有 `model.safetensors.tmp`：`16,777,216` bytes（16 MiB）；其余为配置、tokenizer 与模型 Python 文件。

## 4. `/root/autodl-tmp` 一级目录

| 一级目录 | 占用 | 用途推断 |
|---|---:|---|
| `.ipynb_checkpoints` | 0（6 bytes） | Jupyter 自动检查点，基本为空 |
| `hf_cache` | 964K（955,583 bytes） | Hugging Face 缓存；含 Grounding DINO tiny |
| `.autodl` | 11M（11,148,715 bytes） | AutoDL 平台内部数据/对象元信息 |
| `models` | 64M（66,511,465 bytes） | 独立检测模型；可见 `rtdetr-l.pt` |
| `.Trash-0` | 5.3G（5,649,379,687 bytes） | 数据盘回收站；按约束保持只读 |
| `HolmesVAU-master` | 245G（262,985,809,506 bytes） | 第三章工程、权重、LoRA checkpoints、实验记录及部分数据 |
| `高架桥数据` | 397G（425,922,008,055 bytes） | 六类交通异常视频、训练 JSONL、OVD 结果、验证扫描与论文案例 |

根目录另有一级文件（不属于“一级目录”）：`ovd_test_v5.log` 4.0K、`sample_videos.py` 8.0K、`ovd_train_v5.log` 12K、`e6_train.log` 636K。
