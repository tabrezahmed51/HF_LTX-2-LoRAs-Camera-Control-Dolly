"""
RIFE v4.25 (IFNet_HDv3) frame interpolation — VERBATIM inference architecture.

Provenance:
  Code:   Practical-RIFE train_log/IFNet_HDv3.py (hzwer) — MIT License
          (only change: warplayer import is local; torch.load uses weights_only)
  Weights: official https://huggingface.co/hzwer/RIFE → RIFE_train_log.zip → flownet.pkl
           Keys are DDP-prefixed ("module.") and converted exactly like the
           author's Model.load_model().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RifeInterpolater", "load_rife"]


def warp(tenInput, flow):
    B, C, H, W = tenInput.shape
    gridY, gridX = torch.meshgrid(
        torch.arange(H, device=tenInput.device, dtype=tenInput.dtype),
        torch.arange(W, device=tenInput.device, dtype=tenInput.dtype),
        indexing="ij",
    )
    grid = torch.stack((gridX, gridY), 2)[None].repeat(B, 1, 1, 1)
    grid = grid + flow.permute(0, 2, 3, 1)
    grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(W - 1, 1) - 1.0
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(H - 1, 1) - 1.0
    return F.grid_sample(tenInput, grid, mode="bilinear",
                         padding_mode="border", align_corners=False)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear",
                                 align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(self, x, timestep=0.5, scale_list=[8, 4, 2, 1], training=False,
                fastmode=True, ensemble=False, **kwargs):
        if not training:
            channel = x.shape[1] // 2
            img0, img1 = x[:, :channel], x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow_list, merged, mask_list = [], [], []
        warped_img0, warped_img1 = img0, img1
        flow = mask = None
        for i, block in enumerate([self.block0, self.block1, self.block2,
                                   self.block3, self.block4]):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0[:, :3], img1[:, :3],
                                                    f0, f1, timestep), 1),
                                         None, scale=scale_list[i])
                if ensemble:
                    print("warning: ensemble is not supported since RIFEv4.21")
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block(torch.cat((warped_img0[:, :3], warped_img1[:, :3],
                                                wf0, wf1, timestep, mask, feat), 1),
                                     flow, scale=scale_list[i])
                mask = m0
                flow = flow + fd
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask = torch.sigmoid(mask)
        merged[4] = warped_img0 * mask + warped_img1 * (1 - mask)
        return flow_list, mask_list[4], merged


class RifeInterpolater:
    """Author-style loader (convert keeps module.* keys, strict) + midpoint API."""

    SCALE_LIST = [8, 4, 2, 1]

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.ifnet = IFNet().to(self.device).eval()
        raw = torch.load(weights_path, map_location="cpu", weights_only=False)
        converted = {
            k.replace("module.", ""): v
            for k, v in raw.items() if "module." in k
        }
        # distillation checkpoints carry teacher/caltime heads we don't ship
        converted = {k: v for k, v in converted.items()
                     if not k.startswith(("teacher.", "caltime."))}
        self.ifnet.load_state_dict(converted, strict=True)  # proves pairing

    @torch.inference_mode()
    def _pad32(self, t):
        h, w = t.shape[-2:]
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        return t, h, w

    @torch.inference_mode()
    def midpoint(self, img0: torch.Tensor, img1: torch.Tensor,
                 timestep: float = 0.5) -> torch.Tensor:
        img0p, h, w = self._pad32(img0)
        img1p, _, _ = self._pad32(img1)
        x = torch.cat([img0p, img1p], 1)
        _, _, merged = self.ifnet(x, timestep=timestep, scale_list=self.SCALE_LIST)
        out = merged[4]
        return out[:, :, :h, :w].clamp(0, 1)


def load_rife(weights_path: str, device: str = "cuda") -> RifeInterpolater:
    return RifeInterpolater(weights_path, device)
