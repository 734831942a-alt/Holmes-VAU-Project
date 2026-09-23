# SPEC-04 交付说明

以 specs/SPEC-04.md 为唯一实现契约，正文 SHA-256 保持 09fba92ad69c1968b706588da089c6ca9ee5ad9c8b7f0490ca89696b73da4988。作者已确认原样复制与冻结状态，详见 contract-source issue。

## 数据与划分

manifest 为 6217 行；split 数量：{'train': 4352, 'val': 933, 'test': 932}。camera_group 与可识别源事件跨 split 数均为 0，test 成员已锁定。
流起点非零 5331；容器时长与末帧归零时间不同 6217；含解码诊断 134。所有 duration 都取实际解码末帧，原始视频未修改。
不确定机位 5539 段：{'cluster_boundary': 2881, 'low_confidence_or_missing_overlay': 2658}。均赋 UNKNOWN_i 并列入可复核列表；不能将输出分组无泄漏解读为这些现实机位均已被确认。

## B0 与验收

冻结推理模块直接复用，eager attention，greedy；prompt/帧数等配置见 configs/b0_selfdata.yaml。运行中的配置和 manifest 摘要见 results/b0/run.json；逐次调用凭据见 attempts.jsonl。解析失败保留 raw_output，不重试、不修正区间。最终数量、完成状态与指标以 results/b0/metrics.json 为准。

8 项行为测试通过，包括要求的三项，以及 test 锁定、源事件隔离、OCR 边界、CLI 续跑不重试和缓存根目录保护。§6 每项机器判定见 results/accept-04.json。

可选 pilot30 未执行：没有人工粗标注；没有生成伪标注或正式 IoU。第三章仓库在开始前已有 9 个未跟踪文件；此项不能按冻结标准判通过，且不得清理原始资产以制造通过。

## 依赖与复现

保护包版本未改变，未安装 flash-attn。新增 Python 依赖：onnxruntime、rapidocr_onnxruntime、shapely、pyclipper、coloredlogs、humanfriendly；新增系统工具：tmux、tesseract-ocr、tesseract-ocr-chi-sim。实际版本见 dependencies.json。完整命令见仓库 RUN.md；长任务均在 tmux 中，日志在 logs/spec04-*.log。

## 本次触发的 issue

- issues/04-camera-uncertainty.md
- issues/04-contract-source.md
- issues/04-decoder-diagnostics.md
- issues/04-frame-count-mismatch.md
- issues/04-ocr-unverified.md
- issues/04-third-chapter-dirty.md
- issues/04-timeline-decode.md
- issues/04-timeline-offsets.md


## 用户停跑后的状态

本次交付为实现与诊断检查点，不是 SPEC-04 全量通过。B0 保留 312/932 条原始结果，313 条尝试凭据；一个被中断的调用不自动重试。用户要求先诊断并停止全量，已遵守。离线统计显示 188 条明确合法秒区间因格式不匹配而未被正式解析，27 条存在预算截断迹象；保留原始配置与指标，不以事后诊断替换 B0 结果。

新增 issue：
- issues/04-b0-parser-format.md
- issues/04-b0-stopped-by-user.md

所有 §6 检查已经运行：数据六项通过；完整 B0 指标与预测覆盖未通过；可选人工 pilot 跳过；第三章既有非空 git 状态未通过。验收总状态为 blocked，不能声称全部完成。
