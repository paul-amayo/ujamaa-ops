"""dino_timm.py — DINO ViT-S/8 for LERF's DINO regulariser without torch.hub's GitHub fetch (ovbench, 2026-11-10).

Published LERF (kerrj/lerf db08d57, lerf/data/utils/dino_extractor.py) builds its DINO model with
    torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
which downloads the facebookresearch/dino repository from GitHub; the A100 may not fetch GitHub repositories. This module gives
the extractor the same network and the same weights: timm 0.6.7's `vit_small_patch8_224_dino` loads the official checkpoint
(dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth, the file dino's hubconf loads for dino_vits8),
and the wrapper exposes the interface of dino's VisionTransformer that ViTExtractor uses: patch_embed.patch_size (int),
patch_embed.proj, blocks[i].attn.{qkv,num_heads}, prepare_tokens with dino's interpolate_pos_encoding, forward -> normed CLS.
`check()` verifies the wrapper against timm's own forward at 224 x 224 (no interpolation) — run it before training."""
import math
import timm
import torch
from torch import nn


class _PatchEmbed(nn.Module):   # dino's PatchEmbed: conv, flatten, transpose; no fixed-size assert (timm's has one)
    def __init__(self, proj, patch_size):
        super().__init__(); self.proj = proj; self.patch_size = patch_size

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoViTS8(nn.Module):
    def __init__(self):
        super().__init__()
        t = timm.create_model('vit_small_patch8_224_dino', pretrained=True)
        self.patch_embed = _PatchEmbed(t.patch_embed.proj, 8)
        self.cls_token, self.pos_embed, self.pos_drop, self.blocks, self.norm = t.cls_token, t.pos_embed, t.pos_drop, t.blocks, t.norm
        self.embed_dim = t.embed_dim

    def interpolate_pos_encoding(self, x, w, h):   # facebookresearch/dino vision_transformer.py, verbatim maths
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), mode='bicubic')
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]


def check(dev='cuda'):
    torch.manual_seed(0); m = DinoViTS8().to(dev).eval(); t = timm.create_model('vit_small_patch8_224_dino', pretrained=True).to(dev).eval()
    x = torch.randn(2, 3, 224, 224, device=dev)
    with torch.no_grad():
        a = m(x); b = t.forward_features(x); b = b[:, 0] if b.dim() == 3 else b
        y = torch.randn(1, 3, 496, 880, device=dev); c = m(y)   # non-square, interpolated pos enc runs
    d = (a - b).abs().max().item(); print(f'[dino_timm] wrapper vs timm forward at 224: max |diff| {d:.2e}; 496x880 ok {tuple(c.shape)}', flush=True)
    return d


if __name__ == '__main__':
    import sys; sys.exit(0 if check() < 1e-4 else 1)
