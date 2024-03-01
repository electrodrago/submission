# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmagic.models.archs import pixel_unshuffle
from mmagic.models.utils import default_init_weights, make_layer
from mmagic.registry import MODELS
from mmagic.models.archs import ResidualBlockNoBN


@MODELS.register_module()
class CleanRRDBNet(BaseModule):
    """New design networks consisting of Residual in Residual Dense Block.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
        upscale_factor (int): Upsampling factor. Support x1, x2 and x4.
            Default: 4.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """
    _supported_upscale_factors = [4]

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32,
                 upscale_factor=4,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if upscale_factor in self._supported_upscale_factors:
            in_channels = in_channels * ((4 // upscale_factor)**2)
        else:
            raise ValueError(f'Unsupported scale factor {upscale_factor}. '
                             f'Currently supported ones are '
                             f'{self._supported_upscale_factors}.')

        self.upscale_factor = upscale_factor
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDB,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        
        # Upsample
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # New components
        self.raw_encoder = ResidualBlocksWithInputConv(3, mid_channels // 8, 5)
        self.cleaned_encoder = ResidualBlocksWithInputConv(3, mid_channels - mid_channels // 8, 5)
        self.img_upsample = TransferUpsampling(6)
        self.dynamic_clean = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, 15),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )

        self.dynamic_clean.requires_grad_(False)

        

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        lq_cleaned = x.detach().clone()

        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            residues = self.dynamic_clean(lq_cleaned)
            lq_cleaned = lq_cleaned + residues

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < 1.0:
                break

        feat_raw = self.raw_encoder(x)
        feat_cleaned = self.cleaned_encoder(lq_cleaned)

        feat = torch.cat([feat_raw, feat_cleaned], dim=1)

        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out * 0.2 + self.img_upsample(lq_cleaned, x)

    def init_weights(self):
        """Init weights for models."""
        if self.init_cfg:
            super().init_weights()
        else:
            # Use smaller std for better stability and performance. We
            # use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
            # Generative Adversarial Networks"
            for m in [
                    self.conv_body, self.conv_up1,
                    self.conv_up2, self.conv_hr, self.conv_last
            ]:
                default_init_weights(m, 0.1)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self, mid_channels=64, growth_channels=32):
        super().__init__()
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i+1}',
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3,
                          1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.init_weights()

    def init_weights(self):
        """Init weights for ResidualDenseBlock.

        Use smaller std for better stability and performance. We empirically
        use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
        Generative Adversarial Networks"
        """
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i+1}'), 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self, mid_channels, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

class TransferUpsampling(nn.Module):
    def __init__(self, channels=6):
        super(TransferUpsampling, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, (1, 1))
        self.conv2 = nn.Conv2d(6, 3, (1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.lka_attn = LargeKernelAttn(channels)

        self.linear_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, lq_clean, lq_raw):
        hr_clean = self.linear_x4(lq_clean)
        hr_raw = self.linear_x4(lq_raw)

        hr_raw = self.lrelu(self.conv1(hr_raw))

        hr_clean = self.lka_attn(torch.cat([hr_clean, hr_raw], dim=1))
        hr_clean = self.lrelu(self.conv2(hr_clean))

        return self.linear_x4(lq_clean) + hr_clean

class LargeKernelAttn(nn.Module):
    def __init__(self,
                 channels):
        super(LargeKernelAttn, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )
        self.dwdconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            padding=9,
            groups=channels,
            dilation=3
        )
        self.pwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

    def forward(self, x):
        weight = self.pwconv(self.dwdconv(self.dwconv(x)))

        return x * weight

class ResidualBlocksWithInputConv(BaseModule):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)
