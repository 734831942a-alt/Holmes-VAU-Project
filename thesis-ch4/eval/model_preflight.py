"""Load the frozen inference implementation, without making any B0 prediction."""
import argparse
import traceback

from data.common import config, write_json, issue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    dc = config(cfg["data_config"])
    try:
        import torch
        from src.model.holmesvau_infer import load_model
        torch.set_num_threads(cfg["torch_threads"])
        model, tokenizer = load_model(cfg["model_path"], cfg["device"])
        result = {"load_ok": True, "model_class": type(model).__name__,
                  "tokenizer_class": type(tokenizer).__name__, "device": str(model.device),
                  "predictions_made": 0}
    except Exception:
        detail = traceback.format_exc()
        issue(dc, "model-load", "冻结模型加载失败", detail + "\n未修改 SPEC-01 加载器或权重；不安装 flash-attn，不升级受保护包。")
        write_json(cfg["preflight_output"], {"load_ok": False, "error": detail, "predictions_made": 0})
        raise
    write_json(cfg["preflight_output"], result)
    print(result)


if __name__ == "__main__":
    main()
