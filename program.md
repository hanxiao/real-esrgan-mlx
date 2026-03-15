# Autoresearch: Real-ESRGAN MLX Speed Optimization

## Objective
Maximize inference speed of Real-ESRGAN on Apple Silicon (MLX) with ZERO quality degradation.

## Constraints (HARD, non-negotiable)
1. **Zero quality loss**: max pixel difference vs reference output must be < 1e-4 (float32). Any experiment that fails quality gate is DISCARDED.
2. **No third-party dependencies**: only mlx, numpy, Pillow. No torch, no opencv, no scipy at runtime.
3. **Pure MLX**: all compute must use mlx operations. No fallback to numpy for inference.
4. **Correctness**: all 5 model variants must still work after changes.

## Workflow (repeat until convergence)

1. **Read results.tsv** (`cat results.tsv`) to review all past experiments - what worked, what didn't, what was discarded.
2. **Think** about what to try next. Consider:
   - mx.compile() on forward pass or sub-functions
   - Operator fusion opportunities
   - Memory layout optimization (NHWC is MLX native)
   - Reducing mx.eval() calls / lazy evaluation batching
   - Float16 throughout (already default, but check for unnecessary casts)
   - Tile processing optimization
   - Conv2d kernel optimization
   - Reducing memory allocations
   - Upsample implementation (nearest neighbor)
   - Pixel shuffle/unshuffle optimization
   - Dense block concatenation optimization (RRDBNet has many cat ops)
   - Skip connection patterns
   - Any MLX-specific tricks (metal shaders, compile options)
3. **Implement** the change in model.py and/or upscale.py
4. **Run benchmark**: `uv run python benchmark.py "short description of change"`
5. **Check result**: if status=KEEP, commit. If status=DISCARD (quality failed), revert.
6. **Commit if KEEP**: `git add -A && git commit -m "experiment: <description>, time_512=Xs"`

## Benchmark
- Primary metric: time_512 (512x512 -> 2048x2048, x4plus RRDBNet-23, fp16, median of 5 runs)
- Secondary metric: time_1024 (1024x1024 -> 4096x4096)
- Quality gate: max_diff < 1e-4 vs reference

## Key Architecture Notes
- RRDBNet: 23 RRDB blocks, each with 3 ResidualDenseBlocks, each with 5 Conv2d + dense cat connections
- The dense connections (torch.cat) are the main bottleneck - lots of memory allocation
- pixel_unshuffle for scale=2, pixel_shuffle for final output
- Nearest neighbor upsample 2x before conv_up1 and conv_up2
- Residual scaling: 0.2 at both RDB and RRDB level
- LeakyReLU(0.2) everywhere

## What NOT to do
- Do not change model weights or architecture (no pruning, no quantization below fp16)
- Do not skip any layers or reduce block count
- Do not approximate any operations
- Do not add dependencies
- Do not modify benchmark.py or reference outputs

## Escape Strategy
If 5 consecutive experiments are DISCARDED or show no improvement, try a fundamentally different approach:
- Profile with mx.metal.start_capture() to find actual bottleneck
- Try restructuring the entire forward pass
- Consider custom Metal kernels via mx.fast
