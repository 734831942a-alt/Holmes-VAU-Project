# SPEC-04-A1 交付说明

312/932 顺序前缀、类别仅二轮 221 + 施工 91、不可外推全 test

A1 七项验收全部通过；B0 完整覆盖仍为 FAIL，complete=false。

Layer A 通过 188/312，parse_fail_rate=0.39743589743589747，format_ok_rate=0。原因码：{"no_interval": 117, "single_point": 5, "multi_interval": 0, "missing_unit": 1, "ambiguous_clock": 1}。

存量 312 条均为 unknown，27 条仅疑似截断；未重推理、未恢复全量、未重试中断调用。账本 313/312/1 分记。13 项 A1 单测（33 条语法样例）和 8 项 SPEC-04 回归测试通过；测试生成全部为人工构造 token 的 mock。

28 项受保护文件哈希与修订前一致。未来记录和首批暂停通过模拟测试，尚未进行真实模型的端到端验证（本任务禁止新推理）。

本轮 issues（均已澄清或修复，历史保留）：
- issues/04a1-contract-location.md
- issues/04a1-frozen-guard.md
- issues/04a1-parser-definition.md
- issues/04a1-token-count-semantics.md

初版开发试算归档见 a1-development/initial-reparse/，不是正式交付指标。最终结果只看 reparse_v2/。
