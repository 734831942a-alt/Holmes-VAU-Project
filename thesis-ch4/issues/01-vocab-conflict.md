# SPEC-01 词表差异记录

来源 commit：`8b98e6cda09d078061eb308d78feec75fa23e6fd`

按 SPEC-01 §3.3 合并三处词表时发现以下不一致/歧义，未自行裁决：

1. `ovd_video_summary.py` 的基础 `TEXT_QUERIES` 包含 `debris`、`road obstacle`、`scattered object`，但不含 SARP 目标词表中的 `road debris`、`road spill`、`trash`。移植后两组均保留。
2. `build_ovd_prompt_zh.py` 的设计说明把“路面障碍物/锥桶”同时对应“占道施工/抛洒物”，形成一对多语义；移植后 `road obstacle` 同时保留在这两类的查询映射中。
3. `car/truck/bus` 可同时作为多车事故、拥堵、异常停车的辅助信号，不能仅凭开放词汇检测结果唯一确定类别。
4. `person` 是 VSCM 上下文词，不直接对应六类中的单一类别。

唯一事实来源为 `src/evidence/queries.py`。其中保留全部源词，并用注释明确上述差异；没有把歧义词强行归并为唯一类别。
