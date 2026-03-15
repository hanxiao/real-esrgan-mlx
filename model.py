"""Pure MLX model definitions for Real-ESRGAN."""

import mlx.core as mx
import mlx.nn as nn


def pixel_unshuffle(x: mx.array, scale: int) -> mx.array:
    """Inverse of pixel shuffle. NHWC format.

    Rearranges spatial pixels into channels:
    (N, H, W, C) -> (N, H//s, W//s, C*s*s)
    """
    n, h, w, c = x.shape
    oh, ow = h // scale, w // scale
    x = x.reshape(n, oh, scale, ow, scale, c)
    x = x.transpose(0, 1, 3, 5, 2, 4)  # (N, oh, ow, C, s, s)
    x = x.reshape(n, oh, ow, c * scale * scale)
    return x


def pixel_shuffle(x: mx.array, scale: int) -> mx.array:
    """Pixel shuffle upsampling. NHWC format.

    Rearranges channels into spatial pixels:
    (N, H, W, C*s*s) -> (N, H*s, W*s, C)
    """
    n, h, w, c = x.shape
    oc = c // (scale * scale)
    x = x.reshape(n, h, w, oc, scale, scale)
    x = x.transpose(0, 1, 4, 2, 5, 3)  # (N, H, s, W, s, oc)
    x = x.reshape(n, h * scale, w * scale, oc)
    return x


def nearest_upsample_2x(x: mx.array) -> mx.array:
    """2x nearest neighbor upsampling. NHWC format."""
    n, h, w, c = x.shape
    # Repeat along height and width
    x = mx.broadcast_to(x[:, :, None, :, None, :], (n, h, 2, w, 2, c))
    x = x.reshape(n, h * 2, w * 2, c)
    return x


def pad_reflect(x: mx.array, pad: tuple) -> mx.array:
    """Reflect padding for NHWC tensor.

    Args:
        pad: (left, right, top, bottom)
    """
    left, right, top, bottom = pad
    if top > 0 or bottom > 0:
        top_pad = x[:, 1:top + 1, :, :][:, ::-1, :, :]
        bottom_pad = x[:, -(bottom + 1):-1, :, :][:, ::-1, :, :]
        x = mx.concatenate([top_pad, x, bottom_pad], axis=1)
    if left > 0 or right > 0:
        left_pad = x[:, :, 1:left + 1, :][:, :, ::-1, :]
        right_pad = x[:, :, -(right + 1):-1, :][:, :, ::-1, :]
        x = mx.concatenate([left_pad, x, right_pad], axis=2)
    return x


class ResidualDenseBlock(nn.Module):
    """5-conv dense block with residual scaling."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = nn.leaky_relu(self.conv1(x), negative_slope=0.2)
        x2 = nn.leaky_relu(self.conv2(mx.concatenate([x, x1], axis=-1)), negative_slope=0.2)
        x3 = nn.leaky_relu(self.conv3(mx.concatenate([x, x1, x2], axis=-1)), negative_slope=0.2)
        x4 = nn.leaky_relu(self.conv4(mx.concatenate([x, x1, x2, x3], axis=-1)), negative_slope=0.2)
        x5 = self.conv5(mx.concatenate([x, x1, x2, x3, x4], axis=-1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block: 3x ResidualDenseBlock."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet architecture for Real-ESRGAN x4plus / x2plus / anime_6B."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        in_ch = num_in_ch
        if scale == 2:
            in_ch = num_in_ch * 4
        elif scale == 1:
            in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, padding=1)
        self.body = [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        if self.scale == 2:
            feat = pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, 4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        # upsample 2x twice = 4x total
        feat = nn.leaky_relu(self.conv_up1(nearest_upsample_2x(feat)), negative_slope=0.2)
        feat = nn.leaky_relu(self.conv_up2(nearest_upsample_2x(feat)), negative_slope=0.2)
        out = self.conv_last(nn.leaky_relu(self.conv_hr(feat), negative_slope=0.2))
        return out


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style SR network with PixelShuffle upsampling."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
    ):
        super().__init__()
        self.upscale = upscale
        self.num_out_ch = num_out_ch

        # Build body: first conv + activation, then num_conv*(conv+activation), then last conv
        self.convs = []
        self.acts = []

        # First conv
        self.convs.append(nn.Conv2d(num_in_ch, num_feat, 3, padding=1))
        self.acts.append(nn.PReLU(num_feat))

        # Body convs
        for _ in range(num_conv):
            self.convs.append(nn.Conv2d(num_feat, num_feat, 3, padding=1))
            self.acts.append(nn.PReLU(num_feat))

        # Last conv (no activation)
        self.convs.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, padding=1))

    def __call__(self, x: mx.array) -> mx.array:
        out = x
        for i in range(len(self.convs) - 1):
            out = self.acts[i](self.convs[i](out))
        # Last conv (no activation)
        out = self.convs[-1](out)

        # Pixel shuffle
        out = pixel_shuffle(out, self.upscale)

        # Add nearest upsampled input (network learns residual)
        n, h, w, c = x.shape
        base = mx.broadcast_to(
            x.reshape(n, h, 1, w, 1, c),
            (n, h, self.upscale, w, self.upscale, c),
        ).reshape(n, h * self.upscale, w * self.upscale, c)
        out = out + base
        return out
