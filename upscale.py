"""Real-ESRGAN inference with pure MLX + Pillow. No torch/opencv at runtime.

Usage:
    python upscale.py input.png -o output.png
    python upscale.py input.png --model x4plus --tile 512
    python upscale.py input.png --model animevideo
"""

import argparse
import math
import os
import sys
import time
import json
from pathlib import Path
from urllib.request import urlretrieve

import mlx.core as mx
import mlx.nn as nn
pass  # mx.load used below
import numpy as np
from PIL import Image

from model import RRDBNet, SRVGGNetCompact, pad_reflect


# Weight download URLs (GitHub releases)
WEIGHT_URLS = {
    "x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "animevideo": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "general": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}

# Model name -> safetensors stem
WEIGHT_STEMS = {
    "x4plus": "RealESRGAN_x4plus",
    "x2plus": "RealESRGAN_x2plus",
    "anime_6B": "RealESRGAN_x4plus_anime_6B",
    "animevideo": "realesr-animevideov3",
    "general": "realesr-general-x4v3",
}

MODEL_CONFIGS = {
    "x4plus": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=4,
        num_feat=64, num_block=23, num_grow_ch=32,
    ),
    "x2plus": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=2,
        num_feat=64, num_block=23, num_grow_ch=32,
    ),
    "anime_6B": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=4,
        num_feat=64, num_block=6, num_grow_ch=32,
    ),
    "animevideo": dict(
        arch="srvgg", num_in_ch=3, num_out_ch=3,
        num_feat=64, num_conv=16, upscale=4,
    ),
    "general": dict(
        arch="srvgg", num_in_ch=3, num_out_ch=3,
        num_feat=64, num_conv=32, upscale=4,
    ),
}


def get_weights_dir() -> Path:
    """Get weights cache directory."""
    d = Path(__file__).parent / "weights"
    d.mkdir(exist_ok=True)
    return d


def download_progress(count, block_size, total_size):
    pct = count * block_size * 100 // total_size
    print(f"\rDownloading: {pct}%", end="", flush=True)


def ensure_weights(model_name: str) -> Path:
    """Download and convert weights if needed. Returns path to .safetensors."""
    weights_dir = get_weights_dir()
    stem = WEIGHT_STEMS[model_name]
    safetensors_path = weights_dir / f"{stem}.safetensors"

    if safetensors_path.exists():
        return safetensors_path

    # Download .pth
    pth_path = weights_dir / f"{stem}.pth"
    if not pth_path.exists():
        url = WEIGHT_URLS[model_name]
        print(f"Downloading {stem} from {url}")
        urlretrieve(url, pth_path, reporthook=download_progress)
        print()

    # Convert using convert.py
    print(f"Converting {pth_path} to MLX format...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "convert.py"),
         str(pth_path), "-o", str(weights_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Conversion failed:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout)

    # Clean up .pth to save space
    pth_path.unlink(missing_ok=True)

    return safetensors_path


def load_model(model_name: str, dtype=mx.float16):
    """Load MLX model with weights."""
    config = MODEL_CONFIGS[model_name]
    arch = config["arch"]
    cfg = {k: v for k, v in config.items() if k != "arch"}

    if arch == "rrdb":
        model = RRDBNet(**cfg)
        scale = cfg["scale"]
    else:
        model = SRVGGNetCompact(**cfg)
        scale = cfg["upscale"]

    # Load weights
    weights_path = ensure_weights(model_name)
    weights = mx.load(str(weights_path))

    # Cast weights to target dtype
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    # Compile the forward pass for fused operations
    model = mx.compile(model)

    return model, scale


def upscale_image(
    model,
    img_array: np.ndarray,
    scale: int,
    tile_size: int = 0,
    tile_pad: int = 32,
    pre_pad: int = 10,
    dtype=mx.float16,
) -> np.ndarray:
    """Upscale a single-channel or RGB image array (H, W, C) in [0, 1] float32."""
    h, w, c = img_array.shape

    # Convert to MLX tensor: (1, H, W, C)
    x = mx.array(img_array[None], dtype=dtype)

    # Pre-pad with reflect (right and bottom only, matching PyTorch reference)
    if pre_pad > 0:
        x = pad_reflect(x, (0, pre_pad, 0, pre_pad))

    # Mod pad for scale=2 (needs input divisible by 2)
    mod_scale = None
    mod_pad_h, mod_pad_w = 0, 0
    if scale == 2:
        mod_scale = 2
    elif scale == 1:
        mod_scale = 4
    if mod_scale is not None:
        _, ph, pw, _ = x.shape
        if ph % mod_scale != 0:
            mod_pad_h = mod_scale - ph % mod_scale
        if pw % mod_scale != 0:
            mod_pad_w = mod_scale - pw % mod_scale
        if mod_pad_h > 0 or mod_pad_w > 0:
            x = pad_reflect(x, (0, mod_pad_w, 0, mod_pad_h))

    if tile_size > 0:
        output = tile_process(model, x, scale, tile_size, tile_pad)
    else:
        output = model(x)

    # Remove mod pad
    if mod_scale is not None and (mod_pad_h > 0 or mod_pad_w > 0):
        oh, ow = output.shape[1], output.shape[2]
        output = output[:, :oh - mod_pad_h * scale, :ow - mod_pad_w * scale, :]

    # Remove pre-pad
    if pre_pad > 0:
        oh, ow = output.shape[1], output.shape[2]
        output = output[:, :oh - pre_pad * scale, :ow - pre_pad * scale, :]

    # Clamp and convert back - single eval at the end
    output = mx.clip(output, 0.0, 1.0)
    mx.eval(output)
    return np.array(output[0], dtype=np.float32)


def tile_process(
    model,
    x: mx.array,
    scale: int,
    tile_size: int,
    tile_pad: int,
) -> mx.array:
    """Process image in tiles to handle large inputs."""
    _, height, width, channel = x.shape
    output_height = height * scale
    output_width = width * scale

    # Allocate output
    output = mx.zeros((1, output_height, output_width, channel), dtype=x.dtype)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    total = tiles_x * tiles_y

    for y in range(tiles_y):
        for x_idx in range(tiles_x):
            tile_num = y * tiles_x + x_idx + 1

            # Input tile coordinates
            ofs_x = x_idx * tile_size
            ofs_y = y * tile_size
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # With padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            # Extract and process tile
            tile = x[:, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad, :]
            output_tile = model(tile)
            mx.eval(output_tile)

            print(f"\rTile {tile_num}/{total}", end="", flush=True)

            # Output coordinates
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # Crop padding from output tile
            out_start_x = (input_start_x - input_start_x_pad) * scale
            out_end_x = out_start_x + input_tile_width * scale
            out_start_y = (input_start_y - input_start_y_pad) * scale
            out_end_y = out_start_y + input_tile_height * scale

            cropped = output_tile[:, out_start_y:out_end_y, out_start_x:out_end_x, :]

            # Place tile in output
            output[:, output_start_y:output_end_y, output_start_x:output_end_x, :] = cropped

    if total > 1:
        print()

    return output


def process_image(
    input_path: str,
    output_path: str,
    model_name: str = "x4plus",
    tile_size: int = 0,
    tile_pad: int = 32,
    pre_pad: int = 10,
    fp32: bool = False,
    alpha_upsampler: str = "realesrgan",
):
    """Full pipeline: load image, upscale, save."""
    dtype = mx.float32 if fp32 else mx.float16

    # Load model
    print(f"Loading model: {model_name}")
    t0 = time.time()
    model, scale = load_model(model_name, dtype=dtype)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load image
    img = Image.open(input_path)
    print(f"Input: {img.size[0]}x{img.size[1]} {img.mode}")

    has_alpha = img.mode == "RGBA"
    if img.mode == "L":
        img = img.convert("RGB")
        was_gray = True
    else:
        was_gray = False

    if has_alpha:
        # Split RGB and alpha
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        alpha_img = a
    else:
        rgb = img.convert("RGB")
        alpha_img = None

    # Convert to float32 array [0, 1]
    rgb_array = np.array(rgb, dtype=np.float32) / 255.0

    # Upscale RGB
    print("Upscaling RGB...")
    t0 = time.time()
    output_rgb = upscale_image(model, rgb_array, scale, tile_size, tile_pad, pre_pad, dtype)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s")

    # Handle alpha channel
    if has_alpha:
        alpha_array = np.array(alpha_img, dtype=np.float32) / 255.0
        alpha_array = np.stack([alpha_array] * 3, axis=-1)  # gray -> 3ch

        if alpha_upsampler == "realesrgan":
            print("Upscaling alpha channel...")
            output_alpha = upscale_image(model, alpha_array, scale, tile_size, tile_pad, pre_pad, dtype)
            output_alpha = output_alpha[:, :, 0]  # take single channel
        else:
            # Bilinear resize
            oh, ow = output_rgb.shape[:2]
            alpha_pil = alpha_img.resize((ow, oh), Image.BILINEAR)
            output_alpha = np.array(alpha_pil, dtype=np.float32) / 255.0

        # Combine RGBA
        output_rgba = np.concatenate([
            output_rgb,
            output_alpha[:, :, None],
        ], axis=-1)
        output_uint8 = np.clip(output_rgba * 255.0, 0, 255).astype(np.uint8)
        result = Image.fromarray(output_uint8, "RGBA")
    else:
        output_uint8 = np.clip(output_rgb * 255.0, 0, 255).astype(np.uint8)
        if was_gray:
            # Convert back to grayscale
            result = Image.fromarray(output_uint8).convert("L")
        else:
            result = Image.fromarray(output_uint8, "RGB")

    # Save
    result.save(output_path)
    out_w, out_h = result.size
    print(f"Output: {out_w}x{out_h} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN upscaling with MLX")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument(
        "--model", default="x4plus",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name (default: x4plus)",
    )
    parser.add_argument("--tile", type=int, default=0, help="Tile size (0=no tiling)")
    parser.add_argument("--tile-pad", type=int, default=32, help="Tile padding (default: 32)")
    parser.add_argument("--pre-pad", type=int, default=10, help="Pre-padding (default: 10)")
    parser.add_argument("--fp32", action="store_true", help="Use float32 instead of float16")
    parser.add_argument(
        "--alpha-upsampler", default="realesrgan",
        choices=["realesrgan", "bilinear"],
        help="Alpha channel upsampler (default: realesrgan)",
    )
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.with_stem(p.stem + f"_out"))

    process_image(
        args.input,
        args.output,
        model_name=args.model,
        tile_size=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        fp32=args.fp32,
        alpha_upsampler=args.alpha_upsampler,
    )


if __name__ == "__main__":
    main()
