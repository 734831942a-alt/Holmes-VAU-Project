# Claude 协作约定

本仓库是 Holmes-VAU 项目的代码和协作材料的共享事实来源。

## 与 Claude 协作

- 在把源代码、SPEC、配置、复现材料、评测摘要或审阅材料提供给 Claude 前，先将其放入本仓库的合适路径并完成 Git 提交与推送。
- 冻结 SPEC 必须原样保存在 `specs/`；若有修订，新增版本或修订记录，不能覆写或静默改变已冻结内容。
- 面向 Claude 的说明、问题描述和小型审阅材料放在仓库根目录的 `claude/`；评测材料应保留脚本、配置、摘要和最小可审阅样例。
- 不提交原始视频、完整数据集、模型权重/检查点、运行缓存、密钥、令牌或其他凭据。对于体积大或受限的产物，提交可复现的命令、摘要、清单和少量脱敏/最小样例。

## 当前基线材料

- `benchmark_ovd_efficiency.py`：OVD 效率评测脚本。
- `run_autodl_baseline_500.sh` 与 `internvl_chat/shell/holmesvau_data_baseline.json`：500-step baseline 训练配置。
- `tools/` 下的案例选择、帧提取和预测校正工具：用于人工审阅和案例分析。
- `bridge_experiment_results/e9_multicar_clean_v1/`：保留该实验的评测摘要和 smoke 样例；完整预测保持在服务器，不作为代码仓库内容。
