# SPEC-02 Batch 1 操作说明

## 1. 可复现抽样

```bash
cd /root/thesis-ch4
/root/miniconda3/envs/holmesvau/bin/python -m src.sample.batch1 \
  --config configs/sampling_batch1.yaml
```

抽样只读取 `/root/autodl-tmp/高架桥数据`；不会修改数据盘文件。

## 2. 启动人工标注服务

生产标注文件初始为空，工具不会调用模型，也不会生成、预填或补全任何 `t_start` / `t_end`。

本机没有 tmux。标注服务会持续超过 5 分钟，因此按 SPEC-02 §6 使用 `setsid` 后台运行并落日志：

```bash
cd /root/thesis-ch4
mkdir -p logs
setsid /root/miniconda3/envs/holmesvau/bin/python -m src.annotate.server \
  --manifest data/manifest_batch1.jsonl \
  --output data/annotations_batch1.jsonl \
  --host 127.0.0.1 \
  --port 8080 \
  > logs/annotate-batch1.log 2>&1 < /dev/null &
echo $! > logs/annotate-batch1.pid
```

## 3. Mac 端口转发

按本机 SSH 配置可运行：

```bash
ssh -N -L 8080:localhost:8080 autodl
```

也可按实际连接信息运行：

```bash
ssh -p <端口> -N -L 8080:localhost:8080 root@<主机>
```

然后浏览器打开 <http://localhost:8080>。

## 4. 标注规则与续标

- 起点、终点和标注者必填；两个标志位必须显式选择。
- 用视频控件逐帧/拖动检查；`[` 标起点、`]` 标终点，也可点击按钮。
- “回放已选区间”只回放人工已填写的区间，不修改时间。
- `annotation_seconds` 从该条进入页面到提交，由服务端自动计算。
- 每次提交立即追加并 `fsync` 到 `data/annotations_batch1.jsonl`。
- 重启服务后，已存在于 JSONL 的 `video_id` 会自动跳过。
- 不得用模型、脚本或批处理生成/预填任何时序边界。

人工标注未达到 manifest 行数前停止；不要执行 SPEC-02 §3.3 或 B0。

---

# SPEC-04：独立的自采数据地基与 B0 行为探针

本节只执行 `specs/SPEC-04.md`，不执行上方 SPEC-02 流程。作者于 2026-09-21 明确确认根目录候选正文作为冻结契约并原样复制到 `specs/`；保留文件头原文，授权记录见 `issues/04-contract-source.md`。没有改动任何 SPEC-01 冻结文件。

## 环境与约定

在新实例 `/root/thesis-ch4` 执行；Python 使用 `/root/miniconda3/envs/holmesvau/bin/python`。所有数据路径、模型参数、ROI、阈值、比例、随机种子和 prompt 来自 YAML。源数据为画像对应的 `video_fixed/`，全程只读。代码中的零/一、数组索引及编辑距离单位步长是结构/公式常量，不是未配置的实验超参数。

`configs/data.yaml` 定义数据和时间轴，`configs/b0_selfdata.yaml` 定义 B0，`configs/test04.yaml` 是测试夹具，`configs/accept04.yaml` 定义验收输入。没有安装 flash-attn，没有升级 torch/transformers/peft/accelerate。额外 OCR 和后台工具依赖的实际版本记录在 `results/spec04/dependencies.json`。

## 先探查，再建表

```bash
cd /root/thesis-ch4
tmux new-session -d -s spec04-probe-reproduce \
  'set -o pipefail; /root/miniconda3/envs/holmesvau/bin/python -u -m data.probe_selfdata --config configs/data.yaml 2>&1 | tee logs/spec04-probe-reproduce.log'
```

探查随机抽取 50 段，读取每帧 PTS、流 `start_time / nb_frames / avg_frame_rate / time_base`，并全量读取视频头；生成 `results/probe-selfdata.md` 和 `data/timeline_report.md`。FFmpeg 4.4 的帧 PTS 字段名是 `pkt_pts`，配置已明确指定；不使用 DTS 或 best-effort 时间戳替代真实 PTS。完整解码成功且有诊断的文件保留 `decode_warning`；无 PTS、非单视频流、非单调时间轴等硬错误记 issue 并停止建表，不跳过文件。

```bash
tmux new-session -d -s spec04-manifest-reproduce \
  'set -o pipefail; /root/miniconda3/envs/holmesvau/bin/python -u -m data.build_manifest --src /root/autodl-tmp/高架桥数据/video_fixed --out data/manifest.jsonl --config configs/data.yaml 2>&1 | tee logs/spec04-manifest-reproduce.log'
```

**时间轴**：逐帧 `(pts − first_frame_pts) × time_base`；duration 为最后解码帧的该值，不额外加一帧显示时长。精确比较容器跨度，因此一帧时长差也会标为 `dur_mismatch`。保存的逐帧 PTS 是重算验收和 B0 输入帧时间的证据。

**相机**：固定 ROI 的 OCR 结果做 NFKC 和去空格，然后精确匹配/编辑距离聚类；链式近邻但两端超阈值的整个边界组件进入 uncertain。组名直接采用归一化 OCR 证据，不从文件名或标注推断机位。低置信度样本逐个赋 `UNKNOWN_i`，人工复核所需的原始 OCR、置信度、视频 ID、ROI、帧号都在 `camera_groups.json` 的 `uncertain` 中。UNKNOWN 分组符合契约的保守回退，但不证明这些视频在真实世界来自不同相机。

探查/时间轴/OCR 中间 JSONL 允许在**同一输入、同一相关配置**下继续。若修改数据、ROI、引擎或时间轴配置，先归档相应缓存后重跑；不要拿旧缓存验证新配置。交付的缓存配置证据见结果目录。

## 防泄漏划分

```bash
/root/miniconda3/envs/holmesvau/bin/python -m data.make_splits \
  --manifest data/manifest.jsonl --out data/splits.json --config configs/data.yaml
```

划分相机与来源事件的连通分量，绝不拆分相机组。来源规则取文件名中**来源类型 + 完全相同的事件记录号**；不能解析时仅按相机组。按配置目标比例最小化整组分配的数量误差。六类数量表在 `results/spec04/split-counts.md`。稀有类为零只记 issue，不自动重新抽样。

`data/test-lock.json` 保存首次 test 名单。再次运行必须保持 test 成员完全一致；新相机/来源约束与锁冲突时停止。只有显式 `--reseed` 才允许解除旧锁；如需改变随机种子，在配置中填写，再显式执行该参数。重新建表后需再执行划分以填入 manifest 的 split 字段。

## B0：一次生成、失败保留

```bash
tmux new-session -d -s spec04-b0-reproduce \
  'set -o pipefail; /root/miniconda3/envs/holmesvau/bin/python -u -m eval.b0_selfdata --manifest data/manifest.jsonl --split test --out results/b0 --config configs/b0_selfdata.yaml --resume 2>&1 | tee logs/spec04-b0-reproduce.log'
```

初次可不带 `--resume`；已有结果必须带。直接调用冻结 `src.model.holmesvau_infer.load_model/generate/uniform_indices`，temperature=0，视频加原始 category 为查询条件，不开启 CoT。固定 prompt 包含首帧归零的采样帧秒数；先验证 Decord 与 ffprobe 帧数一致，实际生成返回帧号也必须一致，否则停止，不能用错误时间戳继续。

输出解析只认配置的 START/END 格式；单个数对须有限且 `0 <= start < end`。范围超出视频不会裁剪，另报 out-of-bounds 数量。format_ok 是原始完整输出的严格格式匹配。解析失败仍写 raw_output、parse_ok=false 与失败原因，不重试、不丢弃。比率分母为已完成并持久化的记录；只有 complete=true 时才覆盖全部 test；时长/起点分布只统计解析成功样本；full_clip 用 YAML 中秒级容差比较 `[0,T]`。

`attempts.jsonl` 在调用模型前持久化，`predictions.jsonl` 在返回后持久化。续跑会校验配置、prompt、manifest 摘要。发现有尝试但无完成行时立即写 issue 并停止，绝不自动重试。每条失败也算已完成。未执行可选 §5.5：没有人工 pilot 标签，因此不生成 pilot30，也不报告正式 IoU。

## 测试与验收

```bash
/root/miniconda3/envs/holmesvau/bin/python -m tests.test_spec04 --config configs/test04.yaml
/root/miniconda3/envs/holmesvau/bin/python -m eval.accept04 --config configs/accept04.yaml
```

验收写 `results/accept-04.json`，按 §6 原顺序逐项给布尔值与证据，同时验证冻结哈希、保护包版本和单测。若第三章仓库既有未跟踪文件仍在，§6 最后一项必须失败，命令退出非零；不删除或提交第三章资产来伪造通过。

只读查看后台进度：`tmux list-sessions`，或读取 `logs/spec04-*.log`。所有超过五分钟的探查、建表和推理均在 tmux 中执行，日志落盘。不要在同一输出目录启动两个写入任务。


## 当前 B0 已停止：仅离线诊断

用户要求先查高解析失败原因，不再跑全量。当前保存 312/932 条，尚有 1 条调用前已记账但没有返回值。不要直接执行上面的 B0 复现命令或删除账本；--resume 会拒绝不确定的未完成尝试。参见 issues/04-b0-stopped-by-user.md。

```bash
/root/miniconda3/envs/holmesvau/bin/python -m eval.diagnose04 --config configs/diagnose04.yaml
```

诊断只读取现有输出并加载 tokenizer，不加载模型、不生成新结果。输出 results/spec04/b0-diagnosis.json 的逐条事后分类不构成 B0 新解析协议，不替换正式指标。188 条有合法秒区间但不符合 START/END 格式；27 条有预算截断迹象，生成接口未保留结束原因，不能确证。详见 issues/04-b0-parser-format.md。继续推理和改变正式解析口径均保持暂停。


# SPEC-04-A1：离线复算与未来生成证据

根目录的 SPEC-04-A1_解析口径修订与截断证据记录.md 已经按用户授权原样复制至 specs/SPEC-04-A1.md，SHA-256 为 c0b1e902ad2783a542b55f86a8c1a39e71527a5d690bdc0386710f315c13f2d4。不会修改 SPEC-01 或原 SPEC-04 正文。

本轮仅执行下面的离线命令（不加载模型）：

```bash
/root/miniconda3/envs/holmesvau/bin/python -m tests.test_spec04_a1 --config configs/test04_a1.yaml
/root/miniconda3/envs/holmesvau/bin/python -m eval.reparse_v2 --config configs/b0_reparse_v2.yaml
/root/miniconda3/envs/holmesvau/bin/python -m eval.accept04_a1 --config configs/b0_reparse_v2.yaml
```

结果目录为 results/spec04/reparse_v2/。predictions.json 使用带 scope_notice 的 JSON 对象保存 records，确保文件头显式说明样本范围；不是原 results/b0/predictions.jsonl 的覆盖版本。metrics、原因码直方图、审计和报告也均带同一声明：312/932 顺序前缀、类别仅二轮 221 + 施工 91、不可外推全 test。

Layer A 与严格 format_ok 分开。规则在 YAML；四类示例必覆盖，显式秒数范围与明确 start/end respectively 表达也依总定义处理。任意两个不关联的时间点不猜成区间。多区间不挑选；缺单位不补；歧义时钟不转换。遇到多个失败条件，原始原因列表全部保留，主原因依 YAML 优先序决定，主原因直方图互斥。Layer B 的逆序结果单列、不交换；零时长进分布，越界保留原值并置 out_of_range，不裁剪。率的分母为 312 条持久化记录，中断调用不计 parse fail。

存量全部 stop_reason=unknown，generated_token_count / generated_token_ids / last_token_is_eos / truncated 均为 null（无权威证据）。仅将匹配原输入哈希的既有 tokenizer 重编码长度用于 truncation_suspected；未把重编码结果当生成 token，未回填停止原因。

未来运行的记录入口是 eval.generation_evidence.generate_with_evidence，复用冻结的 src.model.holmesvau_infer.generate。它只在单次调用期间包装语言模型实例的 generate，参数原样转发，返回后或异常时恢复方法；不修改任何冻结源码。实际返回 token IDs 原样保存，EOS 出现→eos；否则数量等于生效预算→max_new_tokens；否则→other。只有 max_new_tokens 才置 truncated=true。预算仍是 32。

§C 变量口径已明确：input_ids 是返回序列中包含的输入 token 前缀。当前 inputs_embeds-only 路径没有这样的前缀，所以长度为 0；不扣除外层 prompt 长度。记录 input_prefix_source / input_prefix_token_count / output_token_count / eos_token_id 作为额外证据。

未来 B0 入口要求显式 --a1-config，并拒绝将原 results/b0 作为输出目录；本轮没有执行任何未来运行。authoritative 字段会同时写入新运行的 predictions 与同一次调用的 attempts 记录。调用前先记账，生成后先写 token 证据；无 prediction 的调用仍按 interrupted 处理，不自动重试。

未来长任务的 first_batch=8、every=25 及暂停阈值来自 future_monitor 配置，只用于运行监控，不改变解析或生成。首次小批量和后续检查报告原文、解析失败率、格式失败率、真实预算停止率；达到阈值就保存结果、记 issue 并暂停，禁止自动续跑。单测用 9 个虚拟样本验证第 8 个时停止，未加载真实模型。

A1 的七项增补验收可以通过，但 B0 全 test 覆盖仍为 FAIL，complete=false；不要混淆两个结论。账本为 attempted=313、persisted=312、interrupted=1。旧 B0 所有文件、旧配置与 SPEC-01 冻结资产共 28 项逐项哈希核验；基线在 results/spec04/a1-before.json。
