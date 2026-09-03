"""Flash-image assets for the stored-input nodes."""

import argparse
import io
import json
import pathlib

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from common import ROOT, load_cfg


def input_quant(model_path):
    interp = tf.lite.Interpreter(model_path=str(model_path))
    scale, zp = interp.get_input_details()[0]["quantization"]
    return scale, zp


def synthetic_images(n, size, seed=0):
    """Structured synthetic frames (gradient + shapes)"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        base = np.linspace(0, 255, size, dtype=np.uint8)
        img = np.stack(np.meshgrid(base, base)[0:1] * 3, axis=-1).reshape(size, size, 3)
        img = (img * rng.uniform(0.4, 1.0, 3)).astype(np.uint8)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for _ in range(rng.integers(2, 6)):
            x0, y0 = rng.integers(0, size - 20, 2)
            w, h = rng.integers(10, 40, 2)
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            if rng.random() < 0.5:
                draw.ellipse([x0, y0, x0 + w, y0 + h], fill=color)
            else:
                draw.rectangle([x0, y0, x0 + w, y0 + h], fill=color)
        out.append((pil, i % 2))
    return out


def vww_images(ann_file, images_dir, n, size, seed=0):
    from train import load_split  # same repo dir

    items = load_split(ann_file, images_dir)
    rng = np.random.default_rng(seed)
    pos = [it for it in items if it[1] == 1]
    neg = [it for it in items if it[1] == 0]
    pick = lambda pool, k: [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    chosen = pick(pos, n // 2) + pick(neg, n - n // 2)
    return [(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR), y) for p, y in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--vww-ann")
    ap.add_argument("--images-dir")
    args = ap.parse_args()

    cfg = load_cfg()["model"]
    size, n, q = cfg["input_size"], cfg["flash_image_count"], cfg["jpeg_quality"]
    model_path = ROOT / cfg["artifacts_dir"] / "model.tflite"
    assert model_path.exists(), "model.tflite missing - run `make model-dev` (or copy the trained model) first"
    scale, zp = input_quant(model_path)
    if not (abs(scale - 1.0) < 0.02 and zp == -128):
        print(f"NOTE: input quant is (scale={scale}, zp={zp}), not (1,-128); raw assets use the real params")

    if args.dev:
        images, out_dir = synthetic_images(n, size), ROOT / "model" / "assets_dev"
    else:
        assert args.vww_ann and args.images_dir, "--vww-ann and --images-dir required without --dev"
        images, out_dir = vww_images(args.vww_ann, args.images_dir, n, size), ROOT / "model" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"jpeg_quality": q, "input_quant": {"scale": scale, "zero_point": int(zp)},
                "synthetic": bool(args.dev), "images": []}
    for i, (pil, label) in enumerate(images):
        jpg = out_dir / f"img{i:02d}.jpg"
        raw = out_dir / f"img{i:02d}.raw"
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=q)
        jpg.write_bytes(buf.getvalue())
        px = np.asarray(pil, dtype=np.float32)
        q8 = np.clip(np.round(px / scale + zp), -128, 127).astype(np.int8)
        raw.write_bytes(q8.tobytes())
        manifest["images"].append({"jpg": jpg.name, "raw": raw.name, "label": label,
                                   "jpeg_bytes": len(buf.getvalue())})
    sizes = [im["jpeg_bytes"] for im in manifest["images"]]
    manifest["avg_jpeg_bytes"] = int(np.mean(sizes))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(images)} images -> {out_dir}  (raw {size*size*3} B each; JPEG avg {manifest['avg_jpeg_bytes']} B, "
          f"min {min(sizes)}, max {max(sizes)})")


if __name__ == "__main__":
    main()
