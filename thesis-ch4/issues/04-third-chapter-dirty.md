# SPEC-04 第三章仓库初始状态非空
状态：验收阻塞；不修改既有资产。
日期：2026-09-21

## 实测
在任何实现写入之前运行：
```bash
git -C /root/autodl-tmp/HolmesVAU-master status --porcelain
```

输出：
```text
?? benchmark_ovd_efficiency.py
?? bridge_experiment_results/e9_multicar_clean_v1/traffic_eval_baseline_clean350_v1.json
?? bridge_experiment_results/e9_multicar_clean_v1/traffic_test_pred_baseline_clean350_v1.jsonl
?? bridge_experiment_results/e9_multicar_clean_v1/traffic_test_pred_baseline_clean350_v1_smoke.jsonl
?? internvl_chat/shell/holmesvau_data_baseline.json
?? run_autodl_baseline_500.sh
?? tools/correct_predictions_e9_final.csv
?? tools/extract_frames_autodl.py
?? tools/select_representative_cases.py
```

## 影响
根目录 SPEC-04 候选文本 §6 要求该输出为空，目前无法满足。
现有 results/accept-01.json 也记录了同样九个未跟踪文件，说明并非本次创建。
不得通过删除、移动、忽略或提交第三章已有文件来制造干净状态。
若正式契约确认沿用该验收项，必须在 accept-04.json 原样记录实际输出，将该项判为不通过，不能改成“状态未变化即通过”。
此项不阻止独立的只读探查；整体实现仍受 issues/04-contract-source.md 阻塞。
