# SPEC-04-A2 唯一一次全 test 运行

本轮授权对应 run_id `30cbdfa7-cc9c-45f9-97e7-72f060b15d94`。服务器根目录的已授权版本原样归档到 `specs/SPEC-04-A2.md`，SHA-256 为 `2a328ba1d517dfb49ab7a602365a43d5a4c499d79da511ca48770400bc988959`。本地根目录旧的待授权稿未采用、未覆盖。生成参数只读取冻结 `configs/b0_selfdata.yaml`。

本 run 已完成 932/932 并通过验收，本次授权已消耗。不能再次 init、清空结果目录或追加推理。旧 `results/b0/*` 和 A1 全部交付物保持不变。

## 顺序与生成

锁定的 932 个 test ID 不变。按类别与 video_id 排序，类别间轮转；只改变遍历顺序，不按模型输出筛选样本。首 16 条覆盖六类，warmup 属于正式结果。每个视频均新生成，旧 312 仅在生成之后用于文本一致性诊断。旧中断样本不做特殊处理。

复用 `eval/generation_evidence.py` 捕获实际返回 token；复用 `src/model/holmesvau_infer.py` 做 eager、greedy 推理。预算 32、帧 12、prompt/温度/采样/输入大小与冻结配置一致。生成产物只保存 raw_output、实际帧号、权威证据和运行元信息，不替换文本。解析和度量调用已接受的 `eval/reparse_v2.py`，写到独立度量目录。

## 后台任务与中途审看

tmux 会话 `spec04-a2`。日志在本目录 `run.log`，进程退出码在 `run.exit`；进度在 `state.json`。每次调用前写 attempts 并 fsync，返回后立即写 predictions 并 fsync。同一个 worker 文件锁防止并发启动。

检查点是 16、66、116、…、916、932。每个检查点停止继续生成，写 `checkpoints/checkpoint-NNNN.json`，等待审看。抽样固定为当前窗口最先的 8 条，不按输出好坏选样本。必须阅读其原文、A1 解析和权威证据，再执行 review：

```bash
/root/miniconda3/envs/holmesvau/bin/python -m eval.b0_full_test review \
  --config configs/b0_full_test.yaml --checkpoint N \
  --verdict continue --note '逐条审看8条原文及证据后的具体结论'
```

generation_exception_rate>0、evidence_malformed_rate>0、empty_raw_output_rate≥0.50、truncated_rate≥0.95 才触发 A2 数值暂停门。检查率使用当前窗口；异常记录保留，不会通过下个窗口清除异常。parse_fail、format_ok、no_interval 不作暂停门。若原文显示明确区间却被错误拒绝，可用 `--verdict pause` 记录实现错误并停止。检查结果及审看说明在 reviews 中；不能覆盖已有决定或绕过触发的门。

出现早于第 8 个可用输出的异常时，仅保留已有原文，不生成替补样本凑八条，记录无法满足八条审看的实际情况并停止；不得以抽样不足为理由继续异常推理。

## 中断与授权

已持久化的样本算完成。仅同一 run 的干净中断可带 `--resume` 继续剩余未开始样本，不重复完成样本。若有调用前 attempt 而无持久化结果，不能证明未发生推理，必须暂停并记 issue，绝不自动重试。同理，存在实现错误或审看结论为 pause 时不能自动继续。

`n_interrupted` 按 A2 定义表示本 run 尚未完成的所有 test 成员（包括尚未开始），不等同于旧 A1 的单个被打断调用。任何时点均满足 n_generated+n_interrupted=932。生成满 932 后再次调用入口也不会加载模型或追加结果。

## 度量与验收

生成结束并完成最后审看后自动产出：全量/按类 metrics、逐条 parsed、312 条重叠的 reproduced_exact_rate 与全部差异文本、protected-after 和 accept-04-a2。精确复现率只作诊断，不以其高低决定通过。无精定位真值，不新增 IoU。

protected-before 包含 A1 原 28 项、接受提交 f22c24c 的全部变更文件、A2 指定的 A1 交付物、旧 B0 与时间轴/划分输入。生成源码与本 run 配置也在 run.json 中记录哈希；不能在中途改参数。

仅重算报告（不推理）：

```bash
/root/miniconda3/envs/holmesvau/bin/python -m eval.b0_full_test report --config configs/b0_full_test.yaml
```

5 项 A2 单测使用人工构造 token 与 mock 模型，验证冻结阈值、模型表现不暂停、八条原文审看、跨类顺序、干净中断后不重复生成。它们不计入真实运行。
