"""FleetSplit slicer: cut a full-int8 .tflite into head_k/tail_k pairs at single-tensor boundaries"""

import argparse
import copy
import hashlib
import json
import pathlib
import sys

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from common import load_cfg  # noqa: E402

OP_NAMES = {v: k for k, v in vars(schema_fb.BuiltinOperator).items() if not k.startswith("_")}


def load_model_t(path):
    buf = pathlib.Path(path).read_bytes()
    return schema_fb.ModelT.InitFromObj(schema_fb.Model.GetRootAs(buf, 0))


def op_name(model_t, op):
    oc = model_t.operatorCodes[op.opcodeIndex]
    code = max(oc.builtinCode, oc.deprecatedBuiltinCode)
    return OP_NAMES.get(code, f"OP_{code}")


def is_const(model_t, tensor_idx):
    t = model_t.subgraphs[0].tensors[tensor_idx]
    if t.buffer == 0:
        return False
    b = model_t.buffers[t.buffer]
    return b.data is not None and len(b.data) > 0


def crossing_tensors(model_t, cut_after):
    """Non-constant tensors produced by ops[0..cut_after] (or graph inputs) and consumed"""
    sg = model_t.subgraphs[0]
    produced = {int(i) for i in sg.inputs}
    for op in sg.operators[: cut_after + 1]:
        produced.update(int(o) for o in op.outputs)
    consumed_later = {int(t) for t in sg.outputs}
    for op in sg.operators[cut_after + 1 :]:
        consumed_later.update(int(i) for i in op.inputs if i >= 0)
    return sorted(t for t in produced & consumed_later if not is_const(model_t, t))


def tensor_info(model_t, tensor_idx):
    t = model_t.subgraphs[0].tensors[tensor_idx]
    shape = [int(s) for s in t.shape]
    qp = t.quantization
    quant = None
    if qp is not None and qp.scale is not None and len(qp.scale):
        quant = {"scale": float(qp.scale[0]), "zero_point": int(qp.zeroPoint[0])}
    return {"tensor_name": t.name.decode() if isinstance(t.name, bytes) else str(t.name),
            "shape": shape, "tensor_bytes": int(np.prod(shape)), "quant": quant}


def candidates(model_t):
    """All op boundaries where exactly one activation crosses (valid cuts)."""
    sg = model_t.subgraphs[0]
    out = []
    for c in range(len(sg.operators) - 1):
        xs = crossing_tensors(model_t, c)
        if len(xs) == 1:
            info = tensor_info(model_t, xs[0])
            out.append({"boundary_op": c, "op_name": op_name(model_t, sg.operators[c]),
                        "crossing_tensor": xs[0], **info})
    return out


def build_submodel(model_t, op_lo, op_hi, in_tensors, out_tensors):
    """New ModelT containing ops[op_lo:op_hi] of subgraph 0."""
    src = model_t.subgraphs[0]
    ops = src.operators[op_lo:op_hi]

    keep, seen = [], set()
    for op in ops:
        for ti in [int(i) for i in op.inputs] + [int(o) for o in op.outputs]:
            if ti >= 0 and ti not in seen:
                seen.add(ti)
                keep.append(ti)
    for ti in list(in_tensors) + list(out_tensors):
        if ti not in seen:
            seen.add(ti)
            keep.append(ti)
    tmap = {old: new for new, old in enumerate(keep)}

    new = schema_fb.ModelT()
    new.version = 3
    new.description = b"fleetsplit slice"
    new.buffers = [schema_fb.BufferT()]  # index 0 = empty sentinel
    new.operatorCodes = []
    bmap, ocmap = {0: 0}, {}

    sg = schema_fb.SubGraphT()
    sg.name = src.name
    sg.tensors = []
    for old in keep:
        t = copy.deepcopy(src.tensors[old])
        ob = int(t.buffer or 0)
        if ob not in bmap:
            bmap[ob] = len(new.buffers)
            new.buffers.append(copy.deepcopy(model_t.buffers[ob]))
        t.buffer = bmap[ob]
        sg.tensors.append(t)

    sg.operators = []
    for op in ops:
        o = copy.deepcopy(op)
        if op.opcodeIndex not in ocmap:
            ocmap[op.opcodeIndex] = len(new.operatorCodes)
            new.operatorCodes.append(copy.deepcopy(model_t.operatorCodes[op.opcodeIndex]))
        o.opcodeIndex = ocmap[op.opcodeIndex]
        o.inputs = np.array([tmap[int(i)] if i >= 0 else -1 for i in op.inputs], dtype=np.int32)
        o.outputs = np.array([tmap[int(x)] for x in op.outputs], dtype=np.int32)
        sg.operators.append(o)

    sg.inputs = np.array([tmap[int(t)] for t in in_tensors], dtype=np.int32)
    sg.outputs = np.array([tmap[int(t)] for t in out_tensors], dtype=np.int32)
    new.subgraphs = [sg]
    return new


def pack_model(model_t):
    b = flatbuffers.Builder(4 * 1024 * 1024)
    b.Finish(model_t.Pack(b), file_identifier=b"TFL3")
    return bytes(b.Output())


def sanity_load(path):
    """The sliced file must load + allocate in the reference interpreter."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    return i["quantization"], o["quantization"]


def cc_codegen(name, blob, out_dir):
    arr = f"g_head_{name}_tflite"
    lines = [f"// generated by slicer/slice.py - do not edit", '#include "models.h"', "", 'extern "C" {',
             f"__attribute__((aligned(16))) const unsigned char {arr}[] = {{"]
    for i in range(0, len(blob), 12):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in blob[i : i + 12]) + ",")
    lines += ["};", f"const unsigned int {arr}_len = {len(blob)};", "}", ""]
    (out_dir / f"head_{name}.cc").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print candidate boundaries and exit")
    args = ap.parse_args()

    cfg = load_cfg()
    art = ROOT / cfg["model"]["artifacts_dir"]
    model_path = art / "model.tflite"
    assert model_path.exists(), "model.tflite missing - run `make model-dev` first"
    model_t = load_model_t(model_path)
    n_ops = len(model_t.subgraphs[0].operators)

    manifest = ROOT / "model" / "assets_dev" / "manifest.json"
    if not manifest.exists():
        manifest = ROOT / "model" / "assets" / "manifest.json"
    jpeg_ref = json.loads(manifest.read_text())["avg_jpeg_bytes"] if manifest.exists() else None

    if args.list:
        print(f"{n_ops} ops; single-tensor boundaries (pin boundary_op + tensor_name in config.yaml):")
        print(f"{'op':>4} {'after op':<18} {'tensor':<42} {'shape':<18} {'bytes':>7}  vs k0-JPEG")
        for c in candidates(model_t):
            ratio = f"{c['tensor_bytes'] / jpeg_ref:5.2f}x" if jpeg_ref else "  n/a"
            print(f"{c['boundary_op']:>4} {c['op_name']:<18} {c['tensor_name'][:42]:<42} "
                  f"{str(c['shape']):<18} {c['tensor_bytes']:>7}  {ratio}")
        return

    cand_by_op = {c["boundary_op"]: c for c in candidates(model_t)}
    fw_dir = ROOT / "firmware" / "models" / "generated"
    fw_dir.mkdir(parents=True, exist_ok=True)

    full_interp_quant = sanity_load(model_path)
    cuts_out = {"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "model_bytes": model_path.stat().st_size, "n_ops": n_ops,
                "input_quant": {"scale": float(full_interp_quant[0][0]), "zero_point": int(full_interp_quant[0][1])},
                "avg_jpeg_bytes": jpeg_ref, "cuts": []}
    externs, size_macros = [], []

    for cut in cfg["cuts"]:
        name, bop = cut["name"], cut["boundary_op"]
        if bop is None:  # k0: full offload, nothing to slice
            cuts_out["cuts"].append({"name": name, "boundary_op": None, "tensor_bytes": jpeg_ref,
                                     "note": "JPEG on air, full model on server"})
            continue
        assert bop in cand_by_op, f"{name}: op {bop} is not a single-tensor boundary (run --list)"
        info = cand_by_op[bop]
        if cut.get("tensor_name"):
            assert info["tensor_name"] == cut["tensor_name"], \
                f"{name}: boundary op {bop} yields tensor {info['tensor_name']!r}, config says {cut['tensor_name']!r}"

        xt = info["crossing_tensor"]
        sg = model_t.subgraphs[0]
        head = build_submodel(model_t, 0, bop + 1, [int(sg.inputs[0])], [xt])
        tail = build_submodel(model_t, bop + 1, n_ops, [xt], [int(t) for t in sg.outputs])
        head_path, tail_path = art / f"head_{name}.tflite", art / f"tail_{name}.tflite"
        head_blob, tail_blob = pack_model(head), pack_model(tail)
        head_path.write_bytes(head_blob)
        tail_path.write_bytes(tail_blob)

        # Verify the invariant
        (_, h_out) = sanity_load(head_path)
        (t_in, _) = sanity_load(tail_path)
        q = info["quant"]
        for got in (h_out, t_in):
            assert abs(got[0] - q["scale"]) < 1e-12 and int(got[1]) == q["zero_point"], \
                f"{name}: boundary quant params drifted: {got} vs {q}"

        cc_codegen(name, head_blob, fw_dir)
        externs.append(f"extern const unsigned char g_head_{name}_tflite[];\n"
                       f"extern const unsigned int  g_head_{name}_tflite_len;")
        size_macros.append(f"#define FLEET_TENSOR_BYTES_{name.upper()} {info['tensor_bytes']}")
        cuts_out["cuts"].append({"name": name, "boundary_op": bop, **{k: info[k] for k in
                                 ("tensor_name", "shape", "tensor_bytes", "quant")},
                                 "head_ops": bop + 1, "tail_ops": n_ops - bop - 1,
                                 "head_tflite_bytes": len(head_blob), "tail_tflite_bytes": len(tail_blob),
                                 "arena_bytes": None})  # measured on device (RecordingMicroAllocator, gate G1/G2)
        print(f"{name}: cut after op {bop} ({info['op_name']}) -> {info['tensor_name']} "
              f"{info['shape']} = {info['tensor_bytes']} B | head {len(head_blob)} B ({bop + 1} ops), "
              f"tail {len(tail_blob)} B ({n_ops - bop - 1} ops)")

    (art / "cuts.json").write_text(json.dumps(cuts_out, indent=2))
    (fw_dir / "models.h").write_text(
        "// generated by slicer/slice.py - do not edit\n#pragma once\n\n"
        '#ifdef __cplusplus\nextern "C" {\n#endif\n\n' + "\n".join(externs) + "\n\n"
        + "\n".join(size_macros) + "\n\n#ifdef __cplusplus\n}\n#endif\n")
    print(f"wrote {art / 'cuts.json'} and {fw_dir}/")


if __name__ == "__main__":
    main()
