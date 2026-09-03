"""Generate a PLACEHOLDER full-int8 model.tflite on the dev machine."""

import numpy as np
import tensorflow as tf

from common import ROOT, build_model, export_int8_tflite, load_cfg


def main():
    cfg = load_cfg()["model"]
    tf.keras.utils.set_random_seed(0)
    size = cfg["input_size"]
    try:
        model = build_model(size, cfg["width_multiplier"], cfg["num_classes"], pretrained=True)
        src = "imagenet backbone + untrained head"
    except Exception as e:  # no network for the keras weight download
        model = build_model(size, cfg["width_multiplier"], cfg["num_classes"], pretrained=False)
        src = f"random init (imagenet weights unavailable: {e})"

    rng = np.random.default_rng(0)

    def rep():
        for _ in range(100):
            yield [rng.uniform(0.0, 255.0, (1, size, size, 3)).astype(np.float32)]

    out = export_int8_tflite(model, rep, ROOT / cfg["artifacts_dir"] / "model.tflite")
    print(f"wrote {out} ({out.stat().st_size} bytes) [{src}]")


if __name__ == "__main__":
    main()
