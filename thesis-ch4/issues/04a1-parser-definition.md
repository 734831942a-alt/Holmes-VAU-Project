# SPEC-04-A1：Layer A 总定义的实现遗漏已修正

状态：已修复的实现问题；未改变冻结条款、模型输出或原 B0 指标。

首次开发实现把 §A 列举的四类示例当成了封闭白名单，因此离线试算得到 186 条。逐条复核条款后确认，§A 的总定义是“输出中恰好一对以秒为单位的非负十进制数值，可无歧义读作 (start, end)”，并没有写“只允许以下四种字符串”。

两端明确标秒的 `at x seconds to y seconds`，以及明确对应起止的 `start and end times are xs and ys, respectively` 满足总定义，初版却将它们标为 no_interval。这是实现遗漏，不应借白名单额外缩小契约。

修正为通用的显式秒范围与显式 start/end respectively 两类规则，并增加人工构造的正常、逆序、缺单位和无关时间点测试；不从任意两个数字猜区间，不使用类别内容、真值、视频内外范围作筛选。规则仍全部在 YAML 中，保留全部失败记录。验收只检验规则与结果一致，不设置 188 或任何目标通过数。

初版配置、代码与离线试算结果在 `results/spec04/a1-development/initial-reparse/` 留档，明确不是最终 A1 结果。`results/spec04/reparse_v2/` 只保留经过修正并通过测试的交付版本。原 results/b0/*、b0_selfdata.yaml、SPEC-01 文件不变；全程没有模型推理。
