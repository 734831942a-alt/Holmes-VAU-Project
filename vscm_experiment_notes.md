# VSCM Experiment Notes

Date: 2026-05-06
Current version before next change: v3.6
File: ovd_video_summary.py

## Current smoke result
Dataset: /root/autodl-tmp/高架桥数据/vscm_smoke_gt.jsonl
OVD output: /root/autodl-tmp/高架桥数据/vscm_smoke_ovd_v36.jsonl

| class | total | VSCM candidates | triggered | all-rate | candidate-rate |
|---|---:|---:|---:|---:|---:|
| 多车事故 | 60 | 24 | 6 | 10.0% | 25.0% |
| 拥堵 | 30 | 8 | 0 | 0.0% | 0.0% |
| 异常停车 | 30 | 12 | 0 | 0.0% | 0.0% |
| 占道施工 | 30 | 1 | 0 | 0.0% | 0.0% |
| 二轮车辆闯入 | 30 | 12 | 0 | 0.0% | 0.0% |
| 抛洒物 | 30 | 7 | 0 | 0.0% | 0.0% |

## Current trigger structure
- 多车事故 trigger reasons: {'NONE_REASON': 4, 'contact': 2}
- Other classes triggered: 0

Triggered multi-car examples:
1. 多车事故_20240617183445_突发事件_35710647.mp4
   static=4 moving=24 ratio=0.143 proximity=False context={construction vehicle:8, person:14, road barrier:3} queue_like=False
2. 多车事故_20240515075458_突发事件_35591684.mp4
   static=4 moving=13 ratio=0.235 proximity=True contact=12 context={construction vehicle:5, person:21, road barrier:12, traffic cone:1} queue_like=False
3. 多车事故_20240614185057_突发事件_35705422.mp4
   static=2 moving=13 ratio=0.133 proximity=False context={construction vehicle:9, person:13, road barrier:14, traffic cone:1} queue_like=False
4. 多车事故_20240614173402_突发事件_35705258.mp4
   static=4 moving=62 ratio=0.061 proximity=False context={construction vehicle:26, person:13, road barrier:17, traffic cone:3} queue_like=True
5. 多车事故_20240622213630_突发事件_35719202.mp4
   static=3 moving=15 ratio=0.167 proximity=True contact=12 close=12 context={construction vehicle:3, person:15, road barrier:3, traffic cone:1} queue_like=False
6. 多车事故_20240620152734_突发事件_35715280.mp4
   static=2 moving=17 ratio=0.105 proximity=False context={construction vehicle:4, person:37, road barrier:12} queue_like=False

## Interpretation
- v3.6 is clean: no false triggers in congestion/parking/construction/two-wheel/debris on this smoke set.
- Multi-car accident recall is still limited: 6/24 candidates, 6/60 all samples.
- The current bottleneck is partly VSCM candidate coverage (24/60), and partly queue_like suppression.
- Queue-scene branch recovered one true multi-car accident but is still strict.

## Planned next change: v3.7
Only relax queue_scene_high_conf, do not touch motorcycle suppression or base VSCM candidate filters.

Current queue_scene_high_conf:
- queue_like
- person >= 12
- construction_vehicle >= 3
- context_frames >= 8
- 2 <= static_count <= 4
- static_ratio <= 0.16
- moving_count >= 10
- center in [0.20, 0.85]

Planned v3.7:
- person >= 8
- construction_vehicle >= 3
- static_ratio <= 0.20
- keep all other constraints unchanged

Goal:
- Raise 多车事故 candidate-rate toward 30%-40%
- Keep 拥堵/异常停车/二轮车辆闯入 <= 10%, ideally 0%

---

## VSCM v3.7 locked checkpoint
Date: 2026-05-06
File snapshot: ovd_video_summary_v37_locked.py

### Smoke result
OVD output: /root/autodl-tmp/高架桥数据/vscm_smoke_ovd_v37.jsonl

| class | total | VSCM candidates | triggered | all-rate | candidate-rate |
|---|---:|---:|---:|---:|---:|
| 多车事故 | 60 | 24 | 7 | 11.7% | 29.2% |
| 拥堵 | 30 | 8 | 0 | 0.0% | 0.0% |
| 异常停车 | 30 | 12 | 0 | 0.0% | 0.0% |
| 占道施工 | 30 | 1 | 0 | 0.0% | 0.0% |
| 二轮车辆闯入 | 30 | 12 | 0 | 0.0% | 0.0% |
| 抛洒物 | 30 | 7 | 0 | 0.0% | 0.0% |

### Interpretation
- This is the cleanest usable VSCM checkpoint so far.
- Multi-car accident candidate trigger rate is 29.2%, just below 30%.
- All other classes have 0 false trigger on the smoke set.
- If later variants increase false positives or reduce multi-car trigger count, roll back to this version.

### Key v3.7 rules
- scene_only_high_conf:
  - person >= 6
  - construction_vehicle >= 3 OR traffic_cone >= 3 OR road_barrier >= 6
  - context_frames >= 6
  - 2 <= static_count <= 4
  - static_ratio <= 0.20
  - moving_count >= 7
  - not queue_like
  - 0.20 <= cluster_center_x_ratio <= 0.85
- queue_scene_high_conf:
  - queue_like
  - person >= 8
  - construction_vehicle >= 3
  - context_frames >= 8
  - 2 <= static_count <= 4
  - static_ratio <= 0.20
  - moving_count >= 10
  - 0.20 <= cluster_center_x_ratio <= 0.85
- motorcycle suppression enabled:
  - if motorcycle_peak >= 1, VSCM trigger is suppressed.

---

## VSCM current checkpoint before candidate-generation work
Date: 2026-05-07
File snapshot: ovd_video_summary_v38_current_locked.py

### Current full-train audit result
Source OVD summary: traffic_train_ovd_summary_v7.jsonl

| class | total | VSCM candidates | triggered | all-rate | candidate-rate |
|---|---:|---:|---:|---:|---:|
| 多车事故 | 141 | 62 | 10 | 7.1% | 16.1% |
| 拥堵 | 1223 | 462 | 7 | 0.6% | 1.5% |
| 异常停车 | 1093 | 430 | 6 | 0.5% | 1.4% |
| 占道施工 | 1101 | 146 | 1 | 0.1% | 0.7% |
| 二轮车辆闯入 | 1170 | 420 | 0 | 0.0% | 0.0% |
| 抛洒物 | 125 | 35 | 0 | 0.0% | 0.0% |

### Offline threshold sweep conclusion
- Under max false all-rate <= 1% and max false candidate-rate <= 5%, best sweep only improved multi-car accident triggers from 10 to 11.
- It also increased congestion false triggers from 7 to 8.
- Therefore, current threshold-level tuning is near saturation.
- Next improvement target should be VSCM candidate generation, not final trigger thresholds.

### Current rule highlights
- motorcycle suppression enabled: motorcycle_peak >= 1 suppresses VSCM trigger.
- scene_only_high_conf:
  - person >= 6
  - construction_vehicle >= 3 OR traffic_cone >= 3 OR road_barrier >= 6
  - context_frames >= 6
  - 2 <= static_count <= 4
  - static_ratio <= 0.20
  - moving_count >= 7
  - not queue_like
  - 0.20 <= cluster_center_x_ratio <= 0.85
- queue_scene_high_conf:
  - queue_like
  - person >= 8
  - construction_vehicle >= 2
  - context_frames >= 8
  - 2 <= static_count <= 4
  - static_ratio <= 0.20
  - moving_count >= 10
  - 0.20 <= cluster_center_x_ratio <= 0.85

### Rollback
To rollback local code to this checkpoint:
PowerShell:
  Copy-Item .\ovd_video_summary_v38_current_locked.py .\ovd_video_summary.py -Force

Server:
  upload ovd_video_summary_v38_current_locked.py and overwrite /root/autodl-tmp/HolmesVAU-master/ovd_video_summary.py

---

## VSCM candidate audit script
Date: 2026-05-07
Script: audit_vscm_candidates.py

### Purpose
This script audits the existing OVD summary file without reading videos or using GPU.
It separates VSCM failures into:
- vscm_none: the video never became a VSCM candidate.
- vscm_candidate but not triggered: candidate was built, but final VSCM rules rejected it.
- triggered: VSCM hint will be injected by downstream prompt builders.

### Important limitation
For vscm_none rows, the script can only infer approximate causes from aggregate OVD counts.
The current OVD summary does not store per-frame vehicle boxes or early-return reasons from _compute_vscm_v3.
If many multi-car accidents fall into G_likely_track_or_static_filter, the next fix must modify candidate generation and write debug fields inside ovd_video_summary.py.

### Server command
Run after generating traffic_train_ovd_summary_v7.jsonl:

```bash
python audit_vscm_candidates.py \
  --gt /root/autodl-tmp/高架桥数据/traffic_train.jsonl \
  --ovd /root/autodl-tmp/高架桥数据/traffic_train_ovd_summary_v7.jsonl \
  --examples-per-bucket 10 \
  --csv /root/autodl-tmp/高架桥数据/vscm_candidate_audit_train.csv
```

### Expected use
- If most multi-car accident misses are vscm_none, threshold sweeping is not enough.
- If most misses are vscm_candidate but not triggered, tune final VSCM trigger thresholds.
- If vscm_none mostly maps to G_likely_track_or_static_filter, improve track matching/static-candidate generation.
