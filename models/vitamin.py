"""
TODO: FIXME:
/usr/local/lib/python3.9/dist-packages/torch/autograd/__init__.py:251: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 1024, 1, 1], strides() = [1024, 1, 1024, 1024]
bucket_view.sizes() = [256, 1024, 1, 1], strides() = [1024, 1, 1, 1] (Triggered internally at ../torch/csrc/distributed/c10d/reducer.cpp:334.)  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.9/dist-packages/torch/autograd/__init__.py:251: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 1024, 1, 1], strides() = [1024, 1, 1024, 1024]

"""

""" ViTamin

Paper: Designing Scalable Vison Models in the Vision-Language Era

@misc{chen2023designing,
      title={Designing Scalable Vison Models in the Vision-Language Era},
      author={Jieneng Chen and Qihang Yu and Xiaohui Shen and Alan Yuille and Liang-Cheih Chen},
      year={2023},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Based on Apache 2.0 licensed code at https://github.com/ViTamin/ViTamin

Modifications and timm support by Jieneng Chen 2023

Reference:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer_hybrid.py
"""

import math
from dataclasses import dataclass
from functools import partial
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint



import timm
from timm.layers import to_2tuple
from timm.layers.norm_act import _create_act
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq, named_apply
from timm.models._registry import register_model
from timm.models.layers import DropPath
from timm.models.layers import create_conv2d, get_norm_act_layer, get_norm_layer, make_divisible
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn
from timm.models.vision_transformer_hybrid import HybridEmbed

# from utils.config import Args
DropPath.__repr__ = lambda self: f'{type(self).__name__}(...)'


@dataclass
class VitConvCfg:
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs)
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    act_layer: str = 'gelu' # stem & stage 1234
    act_layer1: str = 'gelu' # stage 1234
    act_layer2: str = 'gelu' # stage 1234
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None
    down_shortcut: Optional[bool] = True
    mlp: str = 'mlp'

    def __post_init__(self):
        # mbconv vs convnext blocks have different defaults, set in post_init to avoid explicit config args
        use_mbconv = True
        if not self.norm_layer:
            self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
        if not self.norm_layer_cl and not use_mbconv:
            self.norm_layer_cl = 'layernorm'
        if self.norm_eps is None:
            self.norm_eps = 1e-5 if use_mbconv else 1e-6
        self.downsample_pool_type = self.downsample_pool_type or self.pool_type


@dataclass
class VitCfg:
    # embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
    embed_dim: Tuple[Union[int, Tuple[int, ...]], ...] = (96, 192, 384, 768)
    depths: Tuple[Union[int, Tuple[int, ...]], ...] = (2, 3, 5, 2)
    stem_width: int = 64
    conv_cfg: VitConvCfg = None
    weight_init: str = 'vit_eff'
    head_type: str = ""
    stem_type: str = "stem"
    ln2d_permute: bool = True
    # memory_format: str=""


def _init_conv(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Stem(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            act_layer: str = 'gelu',
            norm_layer: str = 'layernorm2d',
            norm_eps: float = 1e-6,
            bias: bool = True,
    ):
        super().__init__()
        self.grad_checkpointing=False
        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs
        self.conv1 = create_conv2d(in_chs, out_chs, 3, stride=2, bias=bias)
        self.norm1 = norm_act_layer(out_chs)
        self.conv2 = create_conv2d(out_chs, out_chs, 3, stride=1, bias=bias)
        named_apply(_init_conv, self)

    def forward(self, x):
        if self.grad_checkpointing:
            x = checkpoint(self.conv1, x)
            x = self.norm1(x)
            x = checkpoint(self.conv2, x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.conv2(x)

        return x


class Downsample2d(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            pool_type: str = 'avg2',
            bias: bool = True,
    ):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)  # spatial downsample
        x = self.expand(x)  # expand chs
        return x


class StridedConv(nn.Module):
    """ downsample 2d as well
    """
    def __init__(
            self,
            kernel_size=3,
            stride=2,
            padding=1,
            in_chans=3,
            embed_dim=768,
            ln2d_permute=True
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.permute = ln2d_permute
        norm_layer = partial(get_norm_layer('layernorm2d'), eps=1e-6)
        self.norm = norm_layer(in_chans) # affine over C

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        return x


class MbConvLNBlock(nn.Module):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int = 1,
        drop_path: float = 0.,
        kernel_size: int = 3,
        norm_layer: str = 'layernorm2d',
        norm_eps: float = 1e-6,
        act_layer: str = 'gelu',
        expand_ratio: float = 4.0,
    ):
        super(MbConvLNBlock, self).__init__()
        self.stride, self.in_chs, self.out_chs = stride, in_chs, out_chs
        mid_chs = make_divisible(out_chs * expand_ratio)
        prenorm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)

        if stride == 2:
            self.shortcut = Downsample2d(in_chs, out_chs, pool_type='avg', bias=True)
        elif in_chs != out_chs:
            self.shortcut = nn.Conv2d(in_chs, out_chs, 1, bias=True)
        else:
            self.shortcut = nn.Identity()

        self.pre_norm = prenorm_act_layer(in_chs, apply_act=False)
        self.down = nn.Identity()
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=1, bias=True)
        self.act1 = _create_act(act_layer, inplace=True)
        self.act2 = _create_act(act_layer, inplace=True)

        self.conv2_kxk = create_conv2d(mid_chs, mid_chs, kernel_size, stride=stride, dilation=1, groups=mid_chs, bias=True)
        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.pre_norm(x)
        x = self.down(x) # nn.Identity()

        # 1x1 expansion conv & act
        x = self.conv1_1x1(x)
        x = self.act1(x)

        # (strided) depthwise 3x3 conv & act
        x = self.conv2_kxk(x)
        x = self.act2(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut

        return x


class MbConvStages(nn.Module):
    """ MobileConv for stage 1 and stage 2 of ViTamin
    """
    def __init__(
            self,
        cfg: VitCfg,
        img_size: Union[int, Tuple[int, int]] = 224, # place holder
        in_chans: int = 3,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.stem = Stem(
            in_chs=in_chans,
            out_chs=cfg.stem_width,
        )
        stages = []
        self.num_stages = len(cfg.embed_dim)
        for s, dim in enumerate(cfg.embed_dim[:2]): # stage
            blocks = []
            stage_in_chs = cfg.embed_dim[s-1] if s>0 else cfg.stem_width
            for d in range(cfg.depths[s]):
                blocks += [MbConvLNBlock(
                        in_chs = stage_in_chs if d==0 else dim,
                        out_chs = dim,
                        stride = 2 if d == 0 else 1,
                        # cfg = cfg.conv_cfg,
                    )]
            blocks = nn.Sequential(*blocks)
            stages += [blocks]

        self.stages = nn.ModuleList(stages)
        self.pool = StridedConv(
                        stride=2,
                        in_chans=cfg.embed_dim[1],
                        embed_dim=cfg.embed_dim[2]
                    )

    def forward(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for stage in self.stages:
                x = checkpoint_seq(stage, x)
            x = checkpoint(self.pool, x)
        else:
            for stage in self.stages:
                x = stage(x)
            x = self.pool(x)

        return x


class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        act_layer = None,
        drop = 0.0,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer('layernorm'), eps=1e-6)
        self.norm = norm_layer(in_features)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(
        self,
        backbone,
        img_size=256,
        patch_size=1,
        feature_size=None,
        in_chans=3,
        embed_dim=1024,
        bias=True,
        dynamic_img_pad=False,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        with torch.no_grad():
            training = backbone.training
            if training:
                backbone.eval()
            o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
            if isinstance(o, (list, tuple)):
                o = o[-1]  # last feature if backbone outputs list/tuple of features
            feature_size = o.shape[-2:]
            feature_dim = o.shape[1]
            backbone.train(training)

        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Upsample2d(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class InvMbConvLNBlock(nn.Module):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int = 1,
        drop_path: float = 0.,
        kernel_size: int = 3,
        norm_layer: str = 'layernorm2d',
        norm_eps: float = 1e-6,
        act_layer: str = 'gelu',
        expand_ratio: float = 4.0,
    ):
        super().__init__()
        self.stride, self.in_chs, self.out_chs = stride, in_chs, out_chs
        mid_chs = make_divisible(out_chs * expand_ratio)
        prenorm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)

        if stride == 2:
            self.shortcut = Upsample2d(in_chs, out_chs)
        elif in_chs != out_chs:
            self.shortcut = nn.Conv2d(in_chs, out_chs, 1, bias=True)
        else:
            self.shortcut = nn.Identity()

        self.pre_norm = prenorm_act_layer(in_chs, apply_act=False)

        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=1, bias=True)
        self.act1 = _create_act(act_layer, inplace=True)
        self.act2 = _create_act(act_layer, inplace=True)

        self.up = Upsample2d(mid_chs, mid_chs) if stride == 2 else nn.Identity()
        self.conv2_kxk = create_conv2d(mid_chs, mid_chs, kernel_size, stride=1, dilation=1, groups=mid_chs, bias=True)
        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.pre_norm(x)

        # 1x1 expansion conv & act
        x = self.conv1_1x1(x)
        x = self.act1(x)
        x = self.up(x)

        # (strided) depthwise 3x3 conv & act
        x = self.conv2_kxk(x)
        x = self.act2(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut

        return x


class InvStem(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        act_layer: str = 'gelu',
        norm_layer: str = 'layernorm2d',
        norm_eps: float = 1e-6,
        bias: bool = True,
    ):
        super().__init__()
        self.grad_checkpointing=False
        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs
        self.conv1 = Upsample2d(in_chs, in_chs)
        self.norm1 = norm_act_layer(in_chs)
        self.conv2 = create_conv2d(in_chs, out_chs, 3, stride=1, bias=bias)
        named_apply(_init_conv, self)

    def forward(self, x):
        if self.grad_checkpointing:
            x = checkpoint(self.conv1, x)
            x = self.norm1(x)
            x = checkpoint(self.conv2, x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.conv2(x)

        return x


class ViTaminDecoder(nn.Module):
    def __init__(
        self,
        model,
        num_query=0,
        img_size=256,
        drop_path=0.,
        depths=(4, 2),
        grad_ckpt=False,
    ):
        super().__init__()

        self.num_query = num_query
        vit = timm.create_model(
            model,
            fc_norm=False,
            patch_size=1,
            drop_rate=0.0,
            num_classes=0,
            global_pool='',
            pos_embed='none',
            mlp_layer=GeGluMlp,
            class_token=False,
            reg_tokens=num_query,
            img_size=img_size,
            drop_path_rate=drop_path,
        )
        self.blocks = vit.blocks
        self.norm_pre = vit.norm_pre
        self.norm = vit.norm

        embed_dims = {
            'vitamin_base': (768, 256, 128),
            'vitamin_large': (1024, 320, 160)
        }[model]
        self.up_conv1 = Upsample2d(embed_dims[0], embed_dims[1])
        self.up_conv2 = nn.Sequential(*[
            InvMbConvLNBlock(
                in_chs=embed_dims[1],
                out_chs=embed_dims[1],
                stride=2 if d == 0 else 1)
            for d in range(depths[0])]
        )
        self.up_conv3 = nn.Sequential(*[
            InvMbConvLNBlock(
                in_chs=embed_dims[1] if d == 0 else embed_dims[2],
                out_chs=embed_dims[2],
                stride=2 if d == 0 else 1)
            for d in range(depths[1])]
        )
        self.up_conv4 = InvStem(in_chs=embed_dims[2], out_chs=3)

        self.grad_ckpt = grad_ckpt

    def get_last_param(self):
        return self.up_conv4.conv2.weight

    def forward(self, x):
        B, L, C = x.shape
        H = W = int((L-self.num_query) ** 0.5)
        x = self.norm_pre(x)
        if self.grad_ckpt:
            x = checkpoint_seq(self.blocks, x)
            x = x[:, self.num_query:, :]
            x = self.norm(x)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)
            x = checkpoint(self.up_conv1, x)
            x = checkpoint_seq(self.up_conv2, x)
            x = checkpoint_seq(self.up_conv3, x)
        else:
            x = self.blocks(x)
            x = x[:, self.num_query:, :]
            x = self.norm(x)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)
            x = self.up_conv1(x)
            x = self.up_conv2(x)
            x = self.up_conv3(x)
        x = self.up_conv4(x)
        return x


def count_params(model: nn.Module):
    return sum([m.numel() for m in model.parameters()])


def count_stage_params(model: nn.Module, prefix='none'):
    collections = []
    for name, m in model.named_parameters():
        print(name)
        if name.startswith(prefix):
            collections.append(m.numel())
    return sum(collections)
