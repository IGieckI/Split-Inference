"""Train + full-int8 quantize the real VWW MobileNetV2-0.35@96."""

import argparse
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf

from common import ROOT, build_model, export_int8_tflite, load_cfg


def load_split(ann_file, images_dir):
    """VWW annotations are COCO-format"""
    d = json.loads(pathlib.Path(ann_file).read_text())
    person_ids = {c["id"] for c in d["categories"] if c["name"] == "person"}
    label = {}
    for a in d["annotations"]:
        lab = 1 if a["category_id"] in person_ids else 0
        label[a["image_id"]] = max(label.get(a["image_id"], 0), lab)
    images_dir = pathlib.Path(images_dir)
    return [(str(images_dir / im["file_name"]), label.get(im["id"], 0)) for im in d["images"]]


def make_ds(items, size, train, batch, seed=0):
    paths = tf.constant([p for p, _ in items])
    labels = tf.constant([l for _, l in items], tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(items), seed=seed, reshuffle_each_iteration=True)

    def load(p, y):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
        img = tf.image.resize(img, (size, size))  # float32 0..255 (pixel domain)
        if train:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 20.0)
            img = tf.clip_by_value(img, 0.0, 255.0)
        return img, y

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def calibration_items(train_items, n, seed=0):
    """Stratified 50/50 person/no-person representative set."""
    rng = np.random.default_rng(seed)
    pos = [it for it in train_items if it[1] == 1]
    neg = [it for it in train_items if it[1] == 0]
    pick = lambda pool, k: [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    return pick(pos, n // 2) + pick(neg, n - n // 2)


def eval_int8(tflite_path, items, size, limit=None):
    interp = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=4)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    scale, zp = inp["quantization"]
    correct = total = 0
    for path, y in items[:limit]:
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (size, size)).numpy()
        q = np.clip(np.round(img / scale + zp), -128, 127).astype(np.int8)[None]
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out["index"])[0]))
        correct += int(pred == y)
        total += 1
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="dir with all COCO-2014 images")
    ap.add_argument("--vww-ann", required=True, help="dir with instances_train.json / instances_val.json")
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-ft", type=int, default=20)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--eval-limit", type=int, default=None, help="cap val images for the int8 eval loop")
    ap.add_argument("--skip-train", action="store_true", help="quantize+eval from saved model_fp32.keras")
    args = ap.parse_args()

    cfg = load_cfg()["model"]
    size, art = cfg["input_size"], ROOT / cfg["artifacts_dir"]
    art.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(0)

    train_items = load_split(pathlib.Path(args.vww_ann) / "instances_train.json", args.images_dir)
    val_items = load_split(pathlib.Path(args.vww_ann) / "instances_val.json", args.images_dir)
    print(f"train={len(train_items)} val={len(val_items)}")

    fp32_path = art / "model_fp32.keras"
    if args.skip_train:
        model = tf.keras.models.load_model(fp32_path)
    else:
        model = build_model(size, cfg["width_multiplier"], cfg["num_classes"], pretrained=True)
        train_ds = make_ds(train_items, size, train=True, batch=args.batch)
        val_ds = make_ds(val_items, size, train=False, batch=args.batch)
        # Stage 1: train the fresh head on a frozen backbone.
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head)
        # Stage 2: fine-tune everything at a low LR.
        for layer in model.layers:
            layer.trainable = True
        model.compile(tf.keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", ["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_ft)
        model.save(fp32_path)

    calib = calibration_items(train_items, cfg["calibration_images"])

    def rep():
        for path, _ in calib:
            img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            img = tf.image.resize(img, (size, size))
            yield [img.numpy().astype(np.float32)[None]]

    tflite_path = export_int8_tflite(model, rep, art / "model.tflite")
    acc = eval_int8(tflite_path, val_items, size, limit=args.eval_limit)
    gate = cfg["int8_accuracy_gate"]
    metrics = {"int8_val_accuracy": acc, "gate": gate, "passed": acc >= gate,
               "val_images": len(val_items[: args.eval_limit or len(val_items)])}
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    if acc < gate:
        print(f"FAIL: int8 accuracy {acc:.3f} < gate {gate} - fix training before touching hardware (section 11.10-C)")
        sys.exit(1)


if __name__ == "__main__":
    main()
