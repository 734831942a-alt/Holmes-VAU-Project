# SPEC-01 B0 阻塞：HIVAU 原视频缺失

## 结论

当前实例只有 HIVAU-70k 的原始 JSON 标注和 instruction JSONL，没有 B0 所需的 UCF-Crime / XD-Violence 原视频，因此不能合法执行 200 段 HolmesVAU-2B 推理，也不能生成真实的 `results/b0/metrics.json` 与 `predictions.jsonl`。

## 证据

- `HIVAU-70k/videos/ucf-crime/videos` 是 62 字节普通文本，内容为旧机器路径：
  `/home/dancer/MLLM/HolmesVAU_full/HIVAU/videos/ucf-crime/videos`
- `HIVAU-70k/videos/xd-violence/videos` 是 64 字节普通文本，内容为旧机器路径：
  `/home/dancer/MLLM/HolmesVAU_full/HIVAU/videos/xd-violence/videos`
- 两者都不是目录或软链接。
- 对 `/root/autodl-tmp` 全盘按 UCF/XD/HIVAU 名称搜索，只找到上述占位文件、它们在 `.Trash-0` 中的相同副本，以及 JSON 标注；未发现 MP4 或数据归档。
- `src/eval/run_b0.py --config configs/b0.yaml` 在模型加载前按 seed=42 选出 200 个事件，并检查视频路径；当前将明确报 200/200 源视频缺失，避免产生伪结果。

没有用 instruction 中的事件裁剪条目或高架桥短片段替代原视频，因为这会违反 SPEC-01 §3.2A/§3.4 的数据定义。

## 恢复条件

需要把与 `raw_annotations/*.json` 对应的 UCF-Crime 与 XD-Violence 原视频恢复到 `configs/b0.yaml` 的 `video_templates` 所列目录之一，或在配置中添加真实视频布局。恢复后按 §6 用 tmux 后台运行 B0。

## flash-attn 处理记录

权重 `config.json` 请求 `flash_attention_2` 且视觉配置 `use_flash_attn=true`，但环境未安装 flash-attn。按 SPEC-01 §6，`src/model/holmesvau_infer.py` 在加载时显式设置视觉 `use_flash_attn=False`、语言模型 `attn_implementation="eager"`，且向模型构造传入 `use_flash_attn=False`；未安装或编译 flash-attn。
