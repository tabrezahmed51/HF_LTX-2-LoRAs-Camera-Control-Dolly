"""
Real-ESRGAN x2 super-resolution — RRDBNet architecture + loader.

Provenance:
  Architecture: standard RRDBNet (Xintao Wang et al., Real-ESRGAN) as used by
                https://huggingface.co/ai-forever/Real-ESRGAN (Apache-2.0 repo)
  Weights:      RealESRGAN_x2.pth from that repo (raw state_dict)
Tile-safe caller keeps ZeroGPU VRAM bounded.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RRDBNet", "load_esrgan", "upscale_tensor"]


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        return self.rdb3(out) * 0.2 + x


class RRDBNet(nn.Module):
    """Sber variant: x2 checkpoints use pixel-unshuffled input (12ch)."""
    def __init__(self, num_in_ch=12, num_out_ch=3, scale=2,
                 num_feat=64, num_block=23, num_grow_ch=32,
                 has_second_up=False):
        super().__init__()
        self.scale = scale
        self.in_ch = num_in_ch
        self.unshuffle_factor = int((num_in_ch // 3) ** 0.5) if num_in_ch != 3 else 0
        self.has_second_up = has_second_up
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if has_second_up or scale == 4:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        if self.unshuffle_factor:
            x = F.pixel_unshuffle(x, self.unshuffle_factor)
        feat = self.conv_body(self.body(self.conv_first(x))) + x
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if hasattr(self, "conv_up2"):
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_esrgan(weights_path: str, device: str = "cuda") -> RRDBNet:
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "params" in sd:
        sd = sd["params"]
    in_ch = sd["conv_first.weight"].shape[1]          # 12 for x2 pixel-unshuffle
    has_up2 = "conv_up2.weight" in sd
    net = RRDBNet(num_in_ch=in_ch, scale=2, has_second_up=has_up2).to(device).eval()
    net.load_state_dict(sd, strict=True)
    return net


@torch.inference_mode()
def upscale_tensor(net: RRDBNet, img: torch.Tensor, tile: int = 384) -> torch.Tensor:
    """img [B,3,H,W] in [0,1]; tiled to bound VRAM. Returns 2x."""
    b, c, h, w = img.shape
    if h <= tile and w <= tile:
        return net(img).clamp(0, 1)
    out = torch.zeros(b, c, h * 2, w * 2, device=img.device, dtype=torch.float32)
    step = tile // 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            t = img[:, :, y:y + step, x:x + step]
            th, tw = t.shape[2], t.shape[3]
            res = net(t).clamp(0, 1)
            out[:, :, y * 2:y * 2 + th * 2, x * 2:x * 2 + tw * 2] = res
    return out
