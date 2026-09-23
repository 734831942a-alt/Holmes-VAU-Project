# SPEC-04 · 初轮机位 OCR 未验证（引擎替换后已解除）

初轮 Tesseract 及修正 ROI 后的抽样均为 0/50 达到置信度阈值；这不能推出画面没有叠字。裁剪图人工检查确认存在机位文字，初始 ROI 底部不足亦已修正。

使用配置允许的 RapidOCR，固定 ROI 与 0.85 置信度阈值，复查相同随机 50 段，有 27 段达到门槛。未据此修改 B0 prompt、推理或解析。

原 Tesseract 证据保存在 results/spec04/tesseract-probe/；当前原始识别及置信度见 results/spec04/ocr.jsonl。其余低置信度、缺失/多行叠字或聚类边界一律 UNKNOWN_i 并进入 uncertain，不人工猜测相机。
