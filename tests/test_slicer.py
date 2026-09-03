"""Slicer tests."""

import json
import pathlib

import numpy as np
import pytest
import tensorflow as tf

ART = pathlib.Path(__file__).resolve().parents[1] / "model" / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ART / "model.tflite").exists() or not (ART / "cuts.json").exists(),
    reason="model artifacts missing - run `make model-dev slice` first",
)


def _interp(path):
    it = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    it.allocate_tensors()
    return it


def _cuts():
    return [c for c in json.loads((ART / "cuts.json").read_text())["cuts"] if c["boundary_op"] is not None]


def test_configured_cuts_are_single_tensor_boundaries():
    from slicer.slice import crossing_tensors, load_model_t

    model_t = load_model_t(ART / "model.tflite")
    for cut in _cuts():
        xs = crossing_tensors(model_t, cut["boundary_op"])
        assert len(xs) == 1, f"{cut['name']}: {len(xs)} tensors cross the boundary"


def test_boundary_quant_params_identical():
    for cut in _cuts():
        head = _interp(ART / f"head_{cut['name']}.tflite")
        tail = _interp(ART / f"tail_{cut['name']}.tflite")
        hq = head.get_output_details()[0]["quantization"]
        tq = tail.get_input_details()[0]["quantization"]
        assert hq == tq == (cut["quant"]["scale"], cut["quant"]["zero_point"])


@pytest.mark.slow
def test_logit_identity_200_images():
    full = _interp(ART / "model.tflite")
    fin, fout = full.get_input_details()[0], full.get_output_details()[0]
    shape = tuple(fin["shape"])

    pairs = []
    for cut in _cuts():
        head = _interp(ART / f"head_{cut['name']}.tflite")
        tail = _interp(ART / f"tail_{cut['name']}.tflite")
        pairs.append((cut["name"], head, tail))

    rng = np.random.default_rng(42)
    mismatches = []
    for i in range(200):
        x = rng.integers(-128, 128, shape, dtype=np.int8)
        full.set_tensor(fin["index"], x)
        full.invoke()
        ref = full.get_tensor(fout["index"]).copy()
        for name, head, tail in pairs:
            head.set_tensor(head.get_input_details()[0]["index"], x)
            head.invoke()
            act = head.get_tensor(head.get_output_details()[0]["index"])
            tail.set_tensor(tail.get_input_details()[0]["index"], act)
            tail.invoke()
            out = tail.get_tensor(tail.get_output_details()[0]["index"])
            if not np.array_equal(ref, out):
                mismatches.append((name, i))
    assert not mismatches, f"non-identical logits: {mismatches[:10]} (of {len(mismatches)})"
