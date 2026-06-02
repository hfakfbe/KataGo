#!/usr/bin/env python3
import argparse
import json
import os
import sys

import torch

from katago.train import modelconfigs
from katago.train.load_model import load_model


class KataGoOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_spatial, input_global):
        outputs_byheads = self.model(input_spatial, input_global)
        batch_size = input_spatial.shape[0]
        (
            out_policy,
            out_value,
            out_miscvalue,
            out_moremiscvalue,
            out_ownership,
            _out_scoring,
            _out_futurepos,
            _out_seki,
            _out_scorebelief_logprobs,
        ) = self.model.float32ify_single_heads_output(outputs_byheads[0])
        return (
            out_policy.reshape(batch_size, out_policy.shape[1], out_policy.shape[2]).contiguous(),
            out_value.reshape(batch_size, out_value.shape[1]).contiguous(),
            out_miscvalue.reshape(batch_size, out_miscvalue.shape[1]).contiguous(),
            out_moremiscvalue.reshape(batch_size, out_moremiscvalue.shape[1]).contiguous(),
            out_ownership.reshape(batch_size, out_ownership.shape[1], out_ownership.shape[2], out_ownership.shape[3]).contiguous(),
        )


def import_real_onnx(repo_root):
    # The repository has a local "onnx/" source directory. When running from the
    # repo root, that shadows the Python onnx package unless we remove it first.
    cwd = os.path.abspath(os.getcwd())
    cleaned = []
    for path in sys.path:
        abspath = os.path.abspath(path or cwd)
        if abspath == repo_root:
            continue
        cleaned.append(path)
    sys.path[:] = cleaned
    import onnx
    return onnx


def add_metadata(onnx, onnx_path, metadata):
    model = onnx.load(onnx_path)
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = str(key)
        entry.value = str(value)
    onnx.save(model, onnx_path)
    onnx.checker.check_model(onnx_path)


def main():
    parser = argparse.ArgumentParser(description="Export a KataGo PyTorch checkpoint to the ONNX format expected by the ONNX TensorRT backend.")
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-export-file", required=True)
    parser.add_argument("-model-name", required=True)
    parser.add_argument("-pos-len", type=int, default=19)
    parser.add_argument("-opset", type=int, default=18)
    parser.add_argument("-trace-batch-size", type=int, default=2)
    parser.add_argument("-use-swa", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    onnx = import_real_onnx(repo_root)

    model, _swa_model, _other_state = load_model(args.checkpoint, args.use_swa, device="cpu", verbose=True)
    model.eval()

    config = model.config
    pos_len = args.pos_len
    num_spatial = modelconfigs.get_num_bin_input_features(config)
    num_global = modelconfigs.get_num_global_input_features(config)
    if num_spatial != 22:
        raise ValueError(f"Unexpected num_spatial_inputs: {num_spatial}")
    if num_global != 19:
        raise ValueError(f"Unexpected num_global_inputs: {num_global}")

    wrapper = KataGoOnnxWrapper(model).eval()
    if args.trace_batch_size < 2:
        raise ValueError("-trace-batch-size must be at least 2 for dynamic batch export")
    input_spatial = torch.zeros((args.trace_batch_size, num_spatial, pos_len, pos_len), dtype=torch.float32)
    input_spatial[:, 0, :, :] = 1.0
    input_global = torch.zeros((args.trace_batch_size, num_global), dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.export_file)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_spatial, input_global),
            args.export_file,
            input_names=["input_spatial", "input_global"],
            output_names=["out_policy", "out_value", "out_miscvalue", "out_moremiscvalue", "out_ownership"],
            dynamic_shapes=({0: "batch"}, {0: "batch"}),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=True,
        )

    metadata = {
        "modelVersion": config["version"],
        "name": args.model_name,
        "num_spatial_inputs": num_spatial,
        "num_global_inputs": num_global,
        "has_mask": "false",
        "pos_len_x": pos_len,
        "pos_len_y": pos_len,
        "is_qat": "false",
        "is_simplified": "false",
        "is_int8": "false",
        "model_config": json.dumps(config, sort_keys=True, separators=(",", ":")),
    }
    add_metadata(onnx, args.export_file, metadata)
    print(f"Exported ONNX model: {args.export_file}")


if __name__ == "__main__":
    main()
