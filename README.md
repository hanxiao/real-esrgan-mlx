# Real-ESRGAN MLX

Real-ESRGAN image upscaling on Apple Silicon. Pure MLX -- no PyTorch, no OpenCV at runtime.

## Install

```bash
uv sync
```

## Usage

```bash
# 4x upscale (weights auto-download on first run)
uv run python upscale.py input.png -o output.png

# models: x4plus (default), x2plus, anime_6B, animevideo, general
uv run python upscale.py input.png --model animevideo

# tile processing for large images
uv run python upscale.py input.png --tile 512

# fp32 precision
uv run python upscale.py input.png --fp32
```

## Models

| Name | Arch | Scale | Params | Notes |
|------|------|-------|--------|-------|
| x4plus | RRDBNet-23 | 4x | 64MB | best quality |
| x2plus | RRDBNet-23 | 2x | 64MB | 2x upscale |
| anime_6B | RRDBNet-6 | 4x | 17MB | anime, lighter |
| animevideo | SRVGG-16 | 4x | 3MB | anime, fastest |
| general | SRVGG-32 | 4x | 6MB | general, fast |

## Performance (M3 Ultra, fp16)

| Input | Model | Output | Time |
|-------|-------|--------|------|
| 512x512 | x4plus | 2048x2048 | 0.7s |
| 64x64 | animevideo | 256x256 | 0.01s |

Numerically equivalent to PyTorch (max diff < 1e-5).

## Weight conversion

One-time, needs torch:

```bash
uv pip install torch
uv run python convert.py model.pth -o weights/ --verify
```

Weights auto-convert on first run via `upscale.py`.

## License

MIT. Weights from [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) under BSD-3.
