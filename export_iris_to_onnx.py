#!/usr/bin/env python3
"""
Export the IrisIResNet50_MSFF embedding backbone checkpoint to ONNX.

Usage:
  python export_iris_to_onnx.py --ckpt best_model.pth --out iris_iresnet50_msff_embedding.onnx

Optional:
  python export_iris_to_onnx.py --ckpt best_model.pth --out iris.onnx --opset 13 --device cpu
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Model definition
# ---------------------------

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IrisIResNet50_MSFF(nn.Module):
    """IResNet50-style 1-channel iris embedding model with Layer3/Layer4 MSFF."""

    fc_scale = 4 * 8

    def __init__(self, num_features=512, dropout_rate=0.5):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        block = IBasicBlock
        layers = [3, 4, 14, 3]

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)

        self.fusion_conv = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(256, eps=1e-05)
        self.fusion_prelu = nn.PReLU(256)

        self.pool = nn.AdaptiveAvgPool2d((4, 8))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(768 * block.expansion * self.fc_scale, num_features)

        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, self.dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        l3_feat = self.layer3(x)
        l4_feat = self.layer4(l3_feat)
        x = self.bn2(l4_feat)

        l3_down = self.fusion_conv(l3_feat)
        l3_down = self.fusion_bn(l3_down)
        l3_down = self.fusion_prelu(l3_down)

        fused = torch.cat((l3_down, x), dim=1)
        fused = self.pool(fused)
        fused = torch.flatten(fused, 1)

        fused = self.dropout(fused)
        x = self.fc(fused)
        x = self.features(x)
        embeds = F.normalize(x, p=2, dim=1)
        return embeds, l3_feat, l4_feat


class ONNXWrapper(nn.Module):
    """Export only the 512-D embedding output."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        emb, _, _ = self.backbone(x)
        return emb


# ---------------------------
# Helpers
# ---------------------------

def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def get_model_state(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise KeyError("Could not find model weights. Expected key 'model_state_dict' or a raw state_dict.")


def torch_onnx_export_compat(model, dummy_input, path, opset_version):
    kwargs = dict(
        args=dummy_input,
        f=str(path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["iris_polar"],
        output_names=["embedding"],
        dynamic_axes={
            "iris_polar": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )
    try:
        torch.onnx.export(model, **kwargs, dynamo=False)
    except TypeError as e:
        if "dynamo" in str(e):
            torch.onnx.export(model, **kwargs)
        else:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best_model.pth")
    parser.add_argument("--out", default="iris_iresnet50_msff_embedding.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset. Use 13 or 12 for older Jetson/TensorRT.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--check", action="store_true", help="Run ONNX checker and optional ONNXRuntime parity check")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location=device)

    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    polar_h = int(config.get("polar_height", config.get("polar_h", 64)))
    polar_w = int(config.get("polar_width", config.get("polar_w", 512)))
    dropout_rate = float(config.get("dropout_rate", 0.5))

    model = IrisIResNet50_MSFF(num_features=512, dropout_rate=dropout_rate).to(device)
    state = strip_module_prefix(get_model_state(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print("WARNING missing keys:", missing)
    if unexpected:
        print("WARNING unexpected keys:", unexpected)
    if missing or unexpected:
        print("If many keys are listed, the script architecture does not match the checkpoint architecture.")

    model.eval()
    export_model = ONNXWrapper(model).to(device).eval()
    dummy = torch.randn(1, 1, polar_h, polar_w, device=device, dtype=torch.float32)

    print(f"Exporting ONNX: {out_path}")
    print(f"Input shape: 1 x 1 x {polar_h} x {polar_w}; opset={args.opset}; device={device}")

    with torch.no_grad():
        torch_onnx_export_compat(export_model, dummy, out_path, args.opset)

    print(f"Saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    meta = {
        "onnx_path": str(out_path),
        "opset": args.opset,
        "polar_height": polar_h,
        "polar_width": polar_w,
        "norm_mean": config.get("norm_mean", 0.449),
        "norm_std": config.get("norm_std", 0.226),
        "radial_inner": config.get("radial_inner", 0.10),
        "radial_outer": config.get("radial_outer", 0.87),
        "use_angular_mask": config.get("use_angular_mask", True),
        "angular_keep_frac": config.get("angular_keep_frac", None),
        "angular_mask_floor": config.get("angular_mask_floor", None),
        "angular_soft_edge": config.get("angular_soft_edge", None),
        "val_threshold_eer": ckpt.get("val_threshold") if isinstance(ckpt, dict) else None,
        "val_threshold_target_far": ckpt.get("val_threshold_far") if isinstance(ckpt, dict) else None,
        "target_far": ckpt.get("target_far", config.get("target_far", None)) if isinstance(ckpt, dict) else None,
    }
    metadata_path = out_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")

    note_path = out_path.with_suffix(".deployment_note.md")
    note_path.write_text(
        f"""# Iris ONNX deployment note

This ONNX file exports only the embedding network:

    normalized polar iris strip ({polar_h}x{polar_w}) -> 512-D L2-normalized embedding

It does not include camera capture, pupil/iris detection, Daugman unwrapping, radial crop, angular mask, enrollment, or threshold decision.

Preprocessing must match training:

- polar image size: {polar_h} x {polar_w}
- normalization mean/std: {meta['norm_mean']} / {meta['norm_std']}
- radial crop: {meta['radial_inner']} to {meta['radial_outer']}
- angular mask: {meta['use_angular_mask']}
- deployment threshold at target FAR: {meta['val_threshold_target_far']}

Use cosine similarity between L2-normalized embeddings.
""",
        encoding="utf-8",
    )
    print(f"Saved deployment note: {note_path}")

    if args.check:
        try:
            import onnx
            onnx.checker.check_model(onnx.load(str(out_path)))
            print("ONNX checker: PASS")
        except Exception as e:
            print(f"ONNX checker warning: {type(e).__name__}: {e}")

        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
            sess = ort.InferenceSession(str(out_path), providers=providers)
            dummy_np = dummy.detach().cpu().numpy().astype(np.float32)
            ort_out = sess.run(None, {"iris_polar": dummy_np})[0]
            with torch.no_grad():
                pt_out = export_model(dummy).detach().cpu().numpy()
            max_abs_diff = float(np.max(np.abs(ort_out - pt_out)))
            print(f"ONNXRuntime parity max_abs_diff: {max_abs_diff:.6g}")
            print(f"Output norm: {np.linalg.norm(ort_out[0]):.6f}")
        except ImportError:
            print("onnxruntime not installed; parity check skipped")
        except Exception as e:
            print(f"ONNXRuntime parity warning: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
