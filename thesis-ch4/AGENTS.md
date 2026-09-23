# 项目约定（Codex 必读）

## 项目
硕士论文第四章：交通异常事件的时空定位（When/Where）。
backbone 为 HolmesVAU-2B，输入固定相机监控视频，输出异常的时序区间与受影响车道。

## 机器约束（违反会导致实例不可用）
- 系统盘 / 仅剩约 19G。任何大于 1G 的文件不得写入系统盘。
- 大文件一律写入 /root/autodl-tmp/（数据盘）。
- HF_HOME 与 PIP_CACHE_DIR 已指向数据盘，不得修改。
- /root/autodl-tmp 下的既有目录是上一阶段工作的资产：
  只读，不得删除、移动或覆盖任何既有文件。

## 环境
- conda 环境已配置完成并验证可用（具体环境名见 results/inventory.md）。
- 不得重建环境，不得升级 torch / transformers / peft / flash-attn。
  缺包时只 pip install 单个包，并在说明里列出新增依赖。

## 第三章代码（冻结引用）
- 路径见 results/inventory.md，只读。
- 需要复用的模块 copy 到本仓库 src/ 下，文件头注释标明来源路径与 commit。
  不得直接 import 该目录，不得修改它。

## 工作方式
- 实现任务只来自 specs/ 下的 SPEC 文件。没有 SPEC 不要写功能代码。
- 严格按 SPEC 实现，不做 SPEC 未要求的改动，包括"顺手优化"。
- 对 SPEC 中的公式或参数有异议时不要自行修改：
  在 issues/ 下新建 md 说明问题，然后停止该任务。
- 每个 SPEC 完成后运行其验收清单，结果写入 results/accept-NN.json。
- docs/ 目录（若存在）仅供理解背景，不是实现依据，不得据此扩展功能。

## 长任务
超过 5 分钟的任务一律 tmux 后台 + 日志落盘，不得前台阻塞：
  tmux new -d -s <名字> "命令 2>&1 | tee logs/<名字>.log"

## Git
- 数据、权重、视频、npz、checkpoint 一律不进 git。
- results/ 下的 json 与 md 要进 git —— 这是评审方读取结果的唯一渠道。
- 提交信息注明对应的 SPEC 编号。
