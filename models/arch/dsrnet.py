from collections import OrderedDict

import torch
import torch.nn as nn
from models.arch.lrm import LRM


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)


class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)


class MuGIBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r


class FeaturePyramidVGG(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.device = 'cuda'
        self.block5 = DualStreamSeq(
            MuGIBlock(512),
            DualStreamBlock(nn.UpsamplingBilinear2d(scale_factor=2.0)),
        )

        self.block4 = DualStreamSeq(
            MuGIBlock(512)
        )

        self.ch_map4 = DualStreamSeq(
            DualStreamBlock(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
                nn.PixelShuffle(2)),
            MuGIBlock(256)
        )

        self.block3 = DualStreamSeq(
            MuGIBlock(256)
        )

        self.ch_map3 = DualStreamSeq(
            DualStreamBlock(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
                nn.PixelShuffle(2)),
            MuGIBlock(128)
        )

        self.block2 = DualStreamSeq(
            MuGIBlock(128)
        )

        self.ch_map2 = DualStreamSeq(
            DualStreamBlock(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
                nn.PixelShuffle(2)),
            MuGIBlock(64)
        )

        self.block1 = DualStreamSeq(
            MuGIBlock(64),
        )

        self.ch_map1 = DualStreamSeq(
            DualStreamBlock(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)),
            MuGIBlock(128),
            DualStreamBlock(nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1)),
            MuGIBlock(32),
        )

        self.block_intro = DualStreamSeq(
            DualStreamBlock(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)),
            MuGIBlock(32)
        )

        self.ch_map0 = DualStreamBlock(
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, inp, vgg_feats):
        # 64,128,256,512,512
        vf1, vf2, vf3, vf4, vf5 = vgg_feats
        # 512=>512,512=>512
        f5_l, f5_r = self.block5(vf5)
        f4_l, f4_r = self.block4(vf4)
        f4_l, f4_r = self.ch_map4(torch.cat([f5_l, f4_l], dim=1), torch.cat([f5_r, f4_r], dim=1))
        # 256 => 256
        f3_l, f3_r = self.block3(vf3)
        # (256+256,256+256)->(128,128)
        f3_l, f3_r = self.ch_map3(torch.cat([f4_l, f3_l], dim=1), torch.cat([f4_r, f3_r], dim=1))
        # (128+128,128+128)->(64,64)
        f2_l, f2_r = self.block2(vf2)
        f2_l, f2_r = self.ch_map2(torch.cat([f3_l, f2_l], dim=1), torch.cat([f3_r, f2_r], dim=1))
        # (64+64,64+64)->(32,32)
        f1_l, f1_r = self.block1(vf1)
        f1_l, f1_r = self.ch_map1(torch.cat([f2_l, f1_l], dim=1), torch.cat([f2_r, f1_r], dim=1))
        # 64
        f0_l, f0_r = self.block_intro(inp, inp)
        f0_l, f0_r = self.ch_map0(torch.cat([f1_l, f0_l], dim=1), torch.cat([f1_r, f0_r], dim=1))
        return f0_l, f0_r


class DSRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, width=48, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = FeaturePyramidVGG(out_channels=width)
        self.ending = DualStreamBlock(nn.Conv2d(width, out_channels, 3, padding=1))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.lrm = LRM(width)

        c = width
        for num in enc_blk_nums:
            self.encoders.append(
                DualStreamSeq(
                    *[MuGIBlock(c) for _ in range(num)]
                )
            )
            self.downs.append(
                DualStreamBlock(
                    nn.Conv2d(c, c * 2, 2, 2)
                )
            )
            c *= 2

        self.middle_blks = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                DualStreamBlock(
                    nn.Conv2d(c, c * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            c //= 2

            self.decoders.append(
                DualStreamSeq(
                    *[MuGIBlock(c) for _ in range(num)]
                )
            )

    def forward(self, inp, feats_inp, fn=None):
        *_, H, W = inp.shape
        x, y = self.intro(inp, feats_inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x, y = encoder(x, y)
            encs.append((x, y))
            x, y = down(x, y)

        x, y = self.middle_blks(x, y)

        for decoder, up, (enc_x_skip, enc_y_skip) in zip(self.decoders, self.ups, encs[::-1]):
            x, y = up(x, y)
            x, y = x + enc_x_skip, y + enc_y_skip
            x, y = decoder(x, y)

        rr = self.lrm(x, y)
        x, y = self.ending(x, y)
        return x, y, rr


if __name__ == '__main__':
    x = torch.ones(1, 3, 224, 224).cuda()
    feats = [
        torch.ones(1, 64, 224, 224).cuda(),
        torch.ones(1, 128, 112, 112).cuda(),
        torch.ones(1, 256, 56, 56).cuda(),
        torch.ones(1, 512, 28, 28).cuda(),
        torch.ones(1, 512, 14, 14).cuda(),
    ]

    enc_blks = [2, 2, 2]
    middle_blk_num = 4
    dec_blks = [2, 2, 2]
    model = DSRNet(3, 3, width=32, middle_blk_num=middle_blk_num,
                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).cuda()
    out_t, out_r, out_rr = model(x, feats)
    print(out_t.shape, out_r.shape, out_rr.shape)
