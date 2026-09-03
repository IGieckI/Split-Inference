"""Shared model construction + int8 export."""

import pathlib
import tempfile

import tensorflow as tf
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def build_model(input_size, alpha, num_classes, pretrained=True):
    """MobileNetV2 backbone with pixel-domain input (0..255)."""
    inp = tf.keras.Input((input_size, input_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
    base = tf.keras.applications.MobileNetV2(
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_tensor=x,
    )
    y = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    y = tf.keras.layers.Dense(num_classes, activation="softmax")(y)
    return tf.keras.Model(inp, y)


def export_int8_tflite(model, representative_gen, out_path):
    """Full-int8 post-training quantization"""
    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp)  # Keras 3 -> SavedModel
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        blob = conv.convert()
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    return out_path
