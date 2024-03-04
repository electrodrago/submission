# Implementation of RealCleanVSR
from logging import WARNING

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint

from mmagic.models.archs import PixelShufflePack, ResidualBlockNoBN
from mmagic.models.utils import flow_warp, make_layer
from mmagic.registry import MODELS
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model.weight_init import constant_init


@MODELS.register_module()
class RealCleanVSRNet(BaseModule):
    """RealCleanVSR architecture

    Args:
        mid_channels (int): Number of mixed channels and propagation channels.
            Default: 64.
        num_blocks (int): Number of channels of the residual blocks in alignment.
            Default: 12.
        num_clean_blocks (int): Number of residual blocks use for Dynamic Clean. 
            Default: 15.
        max_residue_magnitude (int): Maximum of residue magnitude in DCN. 
            Default: 10.
    """
    def __init__(self,
                 mid_channels=64,
                 num_blocks=12,
                 num_clean_blocks=15,
                 max_residue_magnitude=10,
                 spynet_pretrained=None):
        super().__init__()

        self.mid_channels = mid_channels

        # Feature encoder module
        self.raw_encoder = ResidualBlocksWithInputConv(3, mid_channels // 8, 5)
        self.cleaned_encoder = ResidualBlocksWithInputConv(3, mid_channels - mid_channels // 8, 5)

        # Alignment
        self.spynet = SpyNet(pretrained=spynet_pretrained)

        # Deformable alignment and propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_', 'forward_']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        # Reconstruction (Aggregation)
        self.reconstruction = ResidualBlocksWithInputConv(3 * mid_channels, mid_channels, 5)

        # Upsampling
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = TransferUpsampling(6)

        # Activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Dynamic Clean module
        self.dynamic_clean = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_clean_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )

        # Fix SpyNet parameters
        self.spynet.requires_grad_(False)

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        n, t, _, h, w = flows.size()

        # PyTorch 2.0 could not compile data type of 'range'
        # frame_idx = range(0, t + 1)
        # flow_idx = range(-1, t)
        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2)

            # Concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs_clean, lqs_raw, feats):
        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs_clean.size(1)):
            ## Aggregation
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            hr = self.reconstruction(hr)

            # Self-Reference Upsampling
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr += self.img_upsample(lqs_clean[:, i, :, :, :], lqs_raw[:, i, :, :, :])

            # Output
            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs_raw: torch.Tensor, return_lqs=False):
        n, t, c, h, w = lqs_raw.size()

        # Create cleaned image flow path
        lqs_cleaned = lqs_raw.detach().clone()
        
        # Dyna-Mix Cleaning
        ## Dynamic Clean
        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            lqs_cleaned = lqs_cleaned.view(-1, c, h, w)
            residues = self.dynamic_clean(lqs_cleaned)
            lqs_cleaned = (lqs_cleaned + residues).view(n, t, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < 1.0:
                break

        # Compute optical flow
        flows_forward, flows_backward = self.compute_flow(lqs_cleaned)

        feat_raw = self.raw_encoder(lqs_raw.view(-1, c, h, w))
        feat_cleaned = self.cleaned_encoder(lqs_cleaned.view(-1, c, h, w))

        feat_raw = feat_raw.view(n, t, -1, h, w)
        feat_cleaned = feat_cleaned.view(n, t, -1, h, w)
        feats_ = torch.cat([feat_raw, feat_cleaned], dim=2)

        feats = {}
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # Video Refinement stage
        ## Propagation + Alignment
        for direction in ['backward_', 'forward_']:
            feats[direction] = []

            if direction == 'backward_':
                flows = flows_backward
            elif flows_forward is not None:
                flows = flows_forward
            else:
                flows = flows_backward.flip(1)

            feats = self.propagate(feats, flows, direction)
        
        # Aggregation + Upsampling
        if return_lqs:
            return self.upsample(lqs_cleaned, lqs_raw, feats), lqs_cleaned
        else:
            return self.upsample(lqs_cleaned, lqs_raw, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        """Forward function."""
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


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


class LargeKernelAttn(nn.Module):
    """Large Kernel Attention

    Args:
        channels (int): Number of channels
    """
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


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, pretrained=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interpolation='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow = F.interpolate(input=self.process(ref, supp), size=(h, w), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class TransferUpsampling(nn.Module):
    """TransferUpsampling inside Self-Reference Upsampling.

    Args:
        channels (int): Number of channels.
    """
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
