# SPEC-04-A1 §C：output_ids / input_ids 所在接口层不明确

原状态：阻塞（现已澄清，见末尾记录）；仅静态读取本地安装的实现，未进行模型加载或生成。

## 冻结条款

§C 将 `generated_token_count` 写为“新生成 token 数 = len(output_ids) − len(input_ids)”，并要求 `generated_token_ids` 为新生成 token 序列。用户禁止自行修改公式，禁止改 SPEC-01 冻结文件。

## 实际接口

1. 冻结 `src/model/holmesvau_infer.py` 调用 `model.chat()`，只收到字符串和返回帧号，没有生成 token ids。
2. 模型目录的 `modeling_internvl_chat.py` 中，chat 将完整 prompt 的 `input_ids` 交给外层 model.generate。外层 generate 将其转成融合视觉信息的 `inputs_embeds`，调用 `self.language_model.generate(inputs_embeds=..., ...)`；没有把 prompt 的 input_ids 同时传下去。源码约在第 341–353 行。
3. 当前安装的 Transformers 4.46.3，`generation/utils.py` 的 `_maybe_initialize_input_ids_for_generation()` 在 `inputs_embeds` 存在时初始化形状为 `(batch_size, 0)` 的内部 input_ids（约第 564–565 行）。因此此路径的返回 token 序列不包含外层完整 prompt 前缀。

如果把 §C 的 input_ids 理解为 chat 层完整 prompt，直接从该返回序列减去 prompt 长度会得到错误的新生成 token 数，并错误切掉生成 token。不能为了机械套公式而给模型额外传入 input_ids，改变既有生成行为；也不能重编码 raw_output 伪造生成信号。

## 最小待裁决方案

请作者明确 §C 的 input_ids 是否指**实际返回序列所包含的生成输入 token 前缀**：

- 完整 prompt 前缀包含在 output_ids 时，按规定减去该前缀长度；
- 当前 inputs_embeds-only 路径，该有效 token 前缀长度为 0，保留实际返回的全部生成 token ids；外层 prompt 长度可另记证据，不能作为扣除项。

该方案保留“生成 token 数 = 输出序列长度 − 输出中已有输入前缀长度”的公式，但需要作者明确接口层与变量语义后才能落地。尚未采用该解释、未实现替代公式。

可在新增的非冻结包装模块捕获原始生成返回值，复用原冻结推理接口；不能直接修改 SPEC-01 文件。EOS→预算→other 的判定序、32 token 预算、原 B0 输出、313 次尝试账本全部保持不动。后续只能用人工构造的 token 序列做单测，不能为核实此问题实际推理。

本次依据用户第 8 条“若任何条款无法实现或有歧义，停下、写 issues/04a1-*.md，不自行改规则”暂停整个实现；不产出伪装为成功的 reparse_v2 或测试记录。

## 后续澄清与状态

已解决。作者要求明确第 2 点；本轮明确：扣除返回序列实际包含的输入 token 前缀。当前 inputs_embeds-only 路径的有效前缀长度为 0，不减外层 prompt 长度。以新增包装模块记录此证据，冻结文件不变。
