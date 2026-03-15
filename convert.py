"""Convert PyTorch Real-ESRGAN weights to MLX safetensors format.

This script requires torch (one-time conversion tool).
Usage: python convert.py RealESRGAN_x4plus.pth -o weights/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("torch is required for weight conversion. Install it: pip install torch")
    sys.exit(1)

import mlx.core as mx
pass  # mx.save_safetensors used below

from model import RRDBNet, SRVGGNetCompact


# Model configs keyed by filename stem
MODEL_CONFIGS = {
    "RealESRGAN_x4plus": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=4,
        num_feat=64, num_block=23, num_grow_ch=32,
    ),
    "RealESRGAN_x2plus": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=2,
        num_feat=64, num_block=23, num_grow_ch=32,
    ),
    "RealESRGAN_x4plus_anime_6B": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=4,
        num_feat=64, num_block=6, num_grow_ch=32,
    ),
    "realesr-animevideov3": dict(
        arch="srvgg", num_in_ch=3, num_out_ch=3,
        num_feat=64, num_conv=16, upscale=4,
    ),
    "realesr-general-x4v3": dict(
        arch="srvgg", num_in_ch=3, num_out_ch=3,
        num_feat=64, num_conv=32, upscale=4,
    ),
}


def convert_rrdb_weights(state_dict: dict) -> dict:
    """Map PyTorch RRDBNet state_dict keys to MLX weight keys.

    PyTorch conv2d weight: (out_ch, in_ch, kH, kW) - OIHW
    MLX conv2d weight: (out_ch, kH, kW, in_ch) - OHWI
    """
    mlx_weights = {}
    for key, tensor in state_dict.items():
        val = tensor.numpy()

        # Map PyTorch key to MLX key
        # body.0.rdb1.conv1.weight -> body.0.rdb1.conv1.weight
        # The key structure matches, just need to transpose conv weights
        if "weight" in key and val.ndim == 4:
            # OIHW -> OHWI
            val = np.transpose(val, (0, 2, 3, 1))

        mlx_weights[key] = val
    return mlx_weights


def convert_srvgg_weights(state_dict: dict) -> dict:
    """Map PyTorch SRVGGNetCompact state_dict keys to MLX keys.

    PyTorch uses body as ModuleList with alternating conv/activation:
      body.0.weight (conv)
      body.1.weight (prelu)
      body.2.weight (conv)
      body.3.weight (prelu)
      ...
      body.N.weight (last conv, no prelu after)

    MLX uses separate lists for convs and acts:
      convs.0.weight, acts.0.weight,
      convs.1.weight, acts.1.weight,
      ...
      convs.K.weight (last conv)
    """
    mlx_weights = {}
    conv_idx = 0
    act_idx = 0

    # Collect all keys sorted by their index in body
    body_keys = sorted(
        [k for k in state_dict if k.startswith("body.")],
        key=lambda k: int(k.split(".")[1]),
    )

    # Group by body index
    max_idx = max(int(k.split(".")[1]) for k in body_keys)

    for i in range(max_idx + 1):
        sub_keys = [k for k in body_keys if k.startswith(f"body.{i}.")]
        if not sub_keys:
            continue

        # Check if this is a conv or activation
        sample_key = sub_keys[0]
        tensor = state_dict[sample_key]

        if tensor.ndim == 4:
            # Conv2d weight
            for k in sub_keys:
                param_name = k.split(".", 2)[2]  # weight or bias
                val = state_dict[k].numpy()
                if val.ndim == 4:
                    val = np.transpose(val, (0, 2, 3, 1))  # OIHW -> OHWI
                mlx_weights[f"convs.{conv_idx}.{param_name}"] = val
            conv_idx += 1
        elif tensor.ndim == 1:
            # PReLU weight (1D)
            for k in sub_keys:
                param_name = k.split(".", 2)[2]  # weight
                val = state_dict[k].numpy()
                mlx_weights[f"acts.{act_idx}.{param_name}"] = val
            act_idx += 1

    return mlx_weights


def detect_model_config(path: Path, state_dict: dict) -> dict:
    """Detect model config from filename or state_dict structure."""
    stem = path.stem
    if stem in MODEL_CONFIGS:
        return MODEL_CONFIGS[stem]

    # Try to detect from state_dict keys
    if any(k.startswith("body.") and "rdb" in k for k in state_dict):
        # RRDBNet - count blocks
        block_indices = set()
        for k in state_dict:
            if k.startswith("body.") and "rdb" in k:
                block_indices.add(int(k.split(".")[1]))
        num_block = len(block_indices)
        print(f"Detected RRDBNet with {num_block} blocks")
        return dict(
            arch="rrdb", num_in_ch=3, num_out_ch=3, scale=4,
            num_feat=64, num_block=num_block, num_grow_ch=32,
        )
    elif any(k.startswith("body.") for k in state_dict):
        # SRVGGNetCompact - count conv layers
        max_body_idx = max(int(k.split(".")[1]) for k in state_dict if k.startswith("body."))
        # body has: conv, act, conv, act, ..., conv (last)
        # num_conv = (max_body_idx - 2) // 2  # subtract first conv+act and last conv
        num_conv = (max_body_idx - 2) // 2
        print(f"Detected SRVGGNetCompact with {num_conv} conv layers")
        return dict(
            arch="srvgg", num_in_ch=3, num_out_ch=3,
            num_feat=64, num_conv=num_conv, upscale=4,
        )

    raise ValueError(f"Cannot detect model architecture from {path}")


def _build_torch_srvgg(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
    """Build a PyTorch SRVGGNetCompact without importing realesrgan."""
    import torch.nn as tnn
    body = tnn.ModuleList()
    body.append(tnn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
    body.append(tnn.PReLU(num_parameters=num_feat))
    for _ in range(num_conv):
        body.append(tnn.Conv2d(num_feat, num_feat, 3, 1, 1))
        body.append(tnn.PReLU(num_parameters=num_feat))
    body.append(tnn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))

    class _SRVGG(tnn.Module):
        def __init__(self):
            super().__init__()
            self.body = body
            self.upsampler = tnn.PixelShuffle(upscale)
            self.upscale = upscale
        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            base = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode='nearest')
            return out + base

    return _SRVGG()


def _build_torch_rrdb(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
    """Build a PyTorch RRDBNet without importing basicsr."""
    import torch.nn as tnn

    def pixel_unshuffle_torch(x, s):
        b, c, h, w = x.shape
        return x.reshape(b, c, h // s, s, w // s, s).permute(0, 1, 3, 5, 2, 4).reshape(b, c * s * s, h // s, w // s)

    class _RDB(tnn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = tnn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = tnn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = tnn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = tnn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = tnn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = tnn.LeakyReLU(0.2, True)
        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class _RRDB(tnn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = _RDB()
            self.rdb2 = _RDB()
            self.rdb3 = _RDB()
        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    in_ch = num_in_ch
    if scale == 2:
        in_ch = num_in_ch * 4
    elif scale == 1:
        in_ch = num_in_ch * 16

    class _RRDBNet(tnn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            self.conv_first = tnn.Conv2d(in_ch, num_feat, 3, 1, 1)
            self.body = tnn.Sequential(*[_RRDB() for _ in range(num_block)])
            self.conv_body = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = tnn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = tnn.LeakyReLU(0.2, True)
        def forward(self, x):
            if self.scale == 2:
                feat = pixel_unshuffle_torch(x, 2)
            elif self.scale == 1:
                feat = pixel_unshuffle_torch(x, 4)
            else:
                feat = x
            feat = self.conv_first(feat)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return _RRDBNet()


def verify_conversion(mlx_weights: dict, config: dict, state_dict: dict):
    """Verify converted weights produce same output as PyTorch."""
    # Build MLX model
    arch = config.pop("arch") if "arch" in config else config.get("arch")
    cfg = {k: v for k, v in config.items() if k != "arch"}

    if arch == "rrdb":
        mlx_model = RRDBNet(**cfg)
    else:
        mlx_model = SRVGGNetCompact(**cfg)

    # Load weights (ensure mx.array)
    mlx_weights_mx = [(k, mx.array(v) if isinstance(v, np.ndarray) else v) for k, v in mlx_weights.items()]
    mlx_model.load_weights(mlx_weights_mx)

    # Build PyTorch model (inline definitions to avoid basicsr import issues)
    if arch == "rrdb":
        torch_model = _build_torch_rrdb(**cfg)
    else:
        torch_model = _build_torch_srvgg(**cfg)

    torch_model.load_state_dict(state_dict, strict=True)
    torch_model.eval()

    # Random input (small)
    np.random.seed(42)
    inp_np = np.random.rand(1, 64, 64, 3).astype(np.float32)

    # MLX forward (NHWC)
    mlx_out = mlx_model(mx.array(inp_np))
    mx.eval(mlx_out)
    mlx_result = np.array(mlx_out)

    # PyTorch forward (NCHW)
    inp_torch = torch.from_numpy(np.transpose(inp_np, (0, 2, 3, 1)))  # already NHWC
    inp_torch = torch.from_numpy(inp_np.transpose(0, 3, 1, 2))  # NHWC -> NCHW
    with torch.no_grad():
        torch_out = torch_model(inp_torch)
    torch_result = torch_out.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC

    max_diff = np.max(np.abs(mlx_result - torch_result))
    mean_diff = np.mean(np.abs(mlx_result - torch_result))
    print(f"Verification: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    if max_diff > 1e-3:
        print(f"WARNING: max diff {max_diff} exceeds 1e-3 threshold")
    else:
        print("PASS: outputs match within tolerance")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch Real-ESRGAN weights to MLX")
    parser.add_argument("input", type=Path, help="Path to .pth file")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("weights"),
                        help="Output directory for .safetensors")
    parser.add_argument("--verify", action="store_true", help="Verify numerical equivalence")
    args = parser.parse_args()

    # Load PyTorch checkpoint
    print(f"Loading {args.input}...")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)

    # Extract state dict
    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
        print("Using params_ema")
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
        print("Using params")
    else:
        state_dict = checkpoint
        print("Using raw state dict")

    # Detect config
    config = detect_model_config(args.input, state_dict)
    print(f"Config: {config}")

    # Convert
    if config["arch"] == "rrdb":
        mlx_weights = convert_rrdb_weights(state_dict)
    else:
        mlx_weights = convert_srvgg_weights(state_dict)

    # Convert numpy arrays to mx arrays for saving
    mlx_weights_mx = {k: mx.array(v) for k, v in mlx_weights.items()}

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.input.stem}.safetensors"
    mx.save_safetensors(str(out_path), mlx_weights_mx)
    print(f"Saved to {out_path}")

    # Save config as a simple text file for loading
    config_path = args.output_dir / f"{args.input.stem}.config"
    with open(config_path, "w") as f:
        for k, v in config.items():
            f.write(f"{k}={v}\n")
    print(f"Config saved to {config_path}")

    # Verify
    if args.verify:
        print("\nVerifying conversion...")
        verify_conversion(mlx_weights, config.copy(), state_dict)


if __name__ == "__main__":
    main()
