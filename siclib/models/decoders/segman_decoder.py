# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from siclib.models import BaseModel
from siclib.models.utils.functions import resize

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mmcv.cnn import ConvModule
from natten.functional import na2d, na2d_av, na2d_qk
from timm.models.layers import DropPath, to_2tuple

from siclib.models.utils.csm_triton import CrossScanTriton, CrossMergeTriton
import selective_scan_cuda_oflex

from siclib.models.utils.modules import FeatureFusionBlock, FreqFusion, FeatureFusionUpsampleBlock, DySample


#################################################################################
#               Mamba scan functions that preserve image continuity             #
#################################################################################
def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


# fvcore flops =======================================
def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)


    def forward(self, slen):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        return (sin, cos)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5, enable_bias=True):
        super().__init__()
        
        self.dim = dim
        self.init_value = init_value
        self.enable_bias = enable_bias
          
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        if enable_bias:
            self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x
    
    def extra_repr(self) -> str:
        return '{dim}, init_value={init_value}, bias={enable_bias}'.format(**self.__dict__)
    

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels):
        super().__init__(num_groups=1, num_channels=num_channels, eps=1e-6)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


def toodd(size):
    size = to_2tuple(size)
    if size[0] % 2 == 1:
        pass
    else:
        size[0] = size[0] + 1 
    if size[1] % 2 == 1:
        pass
    else:
        size[1] = size[0] + 1
    return size

class VSSM(nn.Module):

    def __init__(
        self,
        d_model=96,
        d_state=1,
        expansion_ratio=1,
        dt_rank="auto",
        norm_layer=LayerNorm2d,
        dropout=0.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        k_groups=4,
        **kwargs,    
    ):
        
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(expansion_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.expansion_ratio = expansion_ratio
        if self.expansion_ratio !=1:
            self.xproj = nn.Linear(d_model, d_inner)
            self.yproj = nn.Linear(d_inner, d_model)
        # # # out proj =======================================
        # self.out_norm = norm_layer(d_inner)
        
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj
        
        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_groups, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_groups, merge=True) # (K * D)
        
        # self.factor1 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
        # self.factor2 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
            
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=None, backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    def _cross_scan(self, x):
        return CrossScanTriton.apply(x)
    
    def _cross_merge(self, x):
        return CrossMergeTriton.apply(x)
    
    
    def forward(self, x, to_dtype=False, force_fp32=False):

        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        
        # xs = torch.stack([x, x.flip([-1])], dim=1).reshape(B, -1, L)
        xs = self._cross_scan(x)
        if self.expansion_ratio!=1:
            xs = self.xproj(xs.permute(0,1,3,2).contiguous()).permute(0,1,3,2).contiguous()
        xs = xs.reshape(B, -1, L)
        x_dbl = F.conv1d(xs, self.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)
              
        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)
                  
        ys = self._selective_scan(xs, dts, As, Bs, 
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)
        
        # y = ys.reshape(B, K, -1, L)
        # yf = F.conv1d(y[:, 0, ...], weight=self.factor1, groups=D)
        # yb = F.conv1d(y[:, 1, ...].flip([-1]), weight=self.factor2, groups=D)
        # y = yf + yb
        y = self._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, H, W)
        
        if self.expansion_ratio!=1:
            y = self.yproj(y.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()

        if to_dtype:
            y = y.to(x.dtype)
        
        return y
     

class Attention(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 window_size, 
                 window_dilation, 
                 global_mode=False, 
                 image_size=None, 
                 use_rpb=False, 
                 sr_ratio=1):
        
        super().__init__()
        window_size = to_2tuple(window_size)
        window_dilation = to_2tuple(window_dilation)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.window_dilation = window_dilation
        self.global_mode = global_mode
        self.sr_ratio = sr_ratio

        image_size=to_2tuple(image_size)
        self.image_size = image_size
        
        self.qkv = nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1)
        self.lepe = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
            
        if use_rpb:
            rpb_list = [nn.Parameter(torch.empty(num_heads, (2 * window_size[0] - 1), (2 * window_size[1] - 1)), requires_grad=True)]
            if global_mode: 
                rpb_list.append(nn.Parameter(torch.empty(num_heads, image_size[0]*image_size[1], image_size[0]*image_size[1]), requires_grad=True))

            self.rpb = nn.ParameterList(rpb_list)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.qkv.weight, gain=2**-2.5)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_normal_(self.proj.weight, gain=2**-2.5)
        nn.init.zeros_(self.proj.bias)
        if hasattr(self, 'rpb'):
            for item in self.rpb:
                nn.init.zeros_(item) # which better? nn.init.trunc_normal_(item, std=0.02)
    
    
    def forward(self, x, pos_enc):
        
        B, C, H, W = x.shape
        
        # attn time
        qkv = self.qkv(x)
        lepe = self.lepe(qkv[:, -C:, ...])
        q, k, v = rearrange(qkv, 'b (m n c) h w -> m b n h w c', m=3, n=self.num_heads)
        
        sin, cos = pos_enc
        q = theta_shift(q, sin, cos) * self.scale
        k = theta_shift(k, sin, cos)
        
        if hasattr(self, 'rpb'):
            rpb = self.rpb[0]
        else:
            rpb = None
              
        attn = na2d_qk(q, k, kernel_size=toodd(self.window_size), dilation=self.window_dilation, rpb=rpb)
        attn = torch.softmax(attn, dim=-1) # b, h, h, w, k^2
        x = na2d_av(attn, v, kernel_size=toodd(self.window_size), dilation=self.window_dilation)

        x = rearrange(x, 'b n_h h w c -> b (n_h c) h w', n_h=self.num_heads).contiguous()
        x = x + lepe
        x = self.proj(x)
        
        return x


class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0,
    ): 
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class VSSMBlock(nn.Module):

    def __init__(self,
                 image_size=None,
                 embed_dim=64,
                 num_heads=2, 
                 expansion_ratio=1,
                 channel_split=False,
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1, 
                 drop_path=0, 
                 layerscale=False, 
                 layer_init_values=1e-6,
                 token_mixer=VSSM,
                 channel_mixer=FFN,
                 norm_layer=LayerNorm2d,
                 dropout=0.1):
        # retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = norm_layer(embed_dim)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.token_mixer = token_mixer(d_model=embed_dim,
                                    k_groups=4, 
                                    expansion_ratio=expansion_ratio,
                                    channel_split=channel_split,)
        self.norm2 = norm_layer(embed_dim)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.mlp = channel_mixer(embed_dim, embed_dim*4)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x):
        x = x + self.cpe1(x)
        token_mix_feat = self.token_mixer(self.norm1(x))
        x = x + self.drop_path(self.layer_scale1(token_mix_feat))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
            
        return x


class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """
    def __init__(self, input_dim=2048, embed_dim=768, identity=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0,2,1).reshape(n, -1, h, w)
        
        return x


class SegMANDecoder(BaseModel):
    default_conf = {
        "predict_uncertainty": True,
        "channels": 144,
        "in_channels": [64, 144, 288, 512],
        "out_channels": 64,
        "in_index": [0, 1, 2, 3],
        "feat_proj_dim": 288,
        "short_cut": False,
        "interpolate_mode": 'bilinear',
        "with_low_level": True,
        "feature_resample": False,
        "feature_resample_group": 4,
        "compress_ratio": 4,
    }

    def _init(self, conf):
        self.in_channels = conf.in_channels
        self.out_channels = conf.out_channels
        self.in_index = conf.in_index
        self.embed_dim = conf.channels
        self.norm_cfg = dict(type='SyncBN', requires_grad=True)
        self.conv_cfg = None
        self.act_cfg = dict(type='ReLU')
        self.align_corners = False
        self.predict_uncertainty = conf.predict_uncertainty
        self.with_ll = conf.with_low_level

        # downsample using convolutions
        self.conv_downsample_2 = ConvModule(
                        self.embed_dim, self.embed_dim*2, kernel_size=3, stride=2, padding=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        self.conv_downsample_4 = ConvModule(
                        self.embed_dim, self.embed_dim*4, kernel_size=5, stride=4, padding=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))

        self.feat_proj_dim = conf.feat_proj_dim

        # try using all features at once
        self.short_cut = conf.short_cut
        self.linear_c4 = MLP(self.in_channels[-1], self.feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], self.feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], self.feat_proj_dim)

        self.feature_resample = conf.feature_resample
        self.feature_resample_group = conf.feature_resample_group
        # self.freqfusion_c4 = FreqFusion(
        #     hr_channels=self.feat_proj_dim, lr_channels=self.feat_proj_dim,
        #     feature_resample=self.feature_resample, feature_resample_group=self.feature_resample_group,
        #     hamming_window=False, compressed_channels=(self.feat_proj_dim * 2) // conf.compress_ratio
        # )
        # self.freqfusion_c3 = FreqFusion(
        #     hr_channels=self.feat_proj_dim, lr_channels=self.feat_proj_dim,
        #     feature_resample=self.feature_resample, feature_resample_group=self.feature_resample_group,
        #     hamming_window=False, compressed_channels=(self.feat_proj_dim * 2) // conf.compress_ratio
        # )
        # self.freqfusion_c2 = FreqFusion(
        #     hr_channels=self.feat_proj_dim, lr_channels=self.feat_proj_dim,
        #     feature_resample=self.feature_resample, feature_resample_group=self.feature_resample_group,
        #     hamming_window=False, compressed_channels=(self.feat_proj_dim * 2) // conf.compress_ratio
        # )

        self.linear_fuse = ConvModule(
                        in_channels=self.feat_proj_dim*3,
                        out_channels=self.embed_dim,
                        kernel_size=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True))
        

        self.reduce_channels = nn.ModuleList([ConvModule(in_channels=self.embed_dim*4*(2**i),
                                out_channels=self.embed_dim,kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True)) for i in range(3)])
        vssm_dim = self.embed_dim*3

        self.vssm =VSSMBlock(embed_dim=vssm_dim,
                                    expansion_ratio=1,
                                    channel_split=False,)

        self.short_path = ConvModule(
                            in_channels=self.embed_dim,
                            out_channels=self.embed_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.image_pool = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1), 
                                ConvModule(self.embed_dim, self.embed_dim, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

        self.proj_out = ConvModule(in_channels=vssm_dim,
                                out_channels=self.feat_proj_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True))

        feat_concat_dim = self.embed_dim*(2+ 3) + self.feat_proj_dim*3
        self.cat = ConvModule(in_channels=feat_concat_dim,
                                out_channels=self.out_channels * 4 * 4,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True)) 

        self.interpolate_mode = conf.interpolate_mode

        if self.predict_uncertainty:
            self.linear_pred_uncertainty = nn.Sequential(
                ConvModule(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.Conv2d(in_channels=self.out_channels, out_channels=1, kernel_size=1),
            )

        if self.with_ll:
            self.out_conv1 = ConvModule(self.out_channels * 4, self.out_channels * 4, 3, padding=1, bias=False)
            self.out_conv2 = ConvModule(self.out_channels, self.out_channels, 3, padding=1, bias=False)
            self.ll_fusion = FeatureFusionUpsampleBlock(self.out_channels, upsample=False)


    def forward_mlp_decoder(self, inputs):
        c1, c2, c3, c4 = inputs

        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)

        _c4 = resize(_c4, size=inputs[1].size()[2:], mode='bilinear', align_corners=False).contiguous()
        _c3 = resize(_c3, size=inputs[1].size()[2:], mode='bilinear', align_corners=False).contiguous()
        # TODO: 采用融合块
        # # c4: 10x10 -> 20x20
        # _, _c3, _c4 = self.freqfusion_c4(hr_feat=_c3, lr_feat=_c4)
        # # c3: 20x20 -> 40x40
        # _, _c2, _c3 = self.freqfusion_c3(hr_feat=_c2, lr_feat=_c3)
        # # c4: 20x20 -> 40x40
        # _, _c2, _c4 = self.freqfusion_c2(hr_feat=_c2, lr_feat=_c4)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        return _c, _c2, _c3, _c4


    def forward_winssm(self, x: torch.Tensor, c2, c3, c4):
        out = [self.short_path(x), 
                  resize(self.image_pool(x),
                        size=x.size()[2:],
                        mode='bilinear',
                        align_corners=self.align_corners).contiguous()]

        B, C, H, W = x.size()

        # obtain multi scale features
        x_2 = self.conv_downsample_2(x) # 1/2 resolution
        x_4 = self.conv_downsample_4(x) # 1/4 resolution

        # unshuffle all features to size 1/4 resolution (16x16 for 512 input res)
        x_2_unshuffle = F.pixel_unshuffle(x_2, downscale_factor=2)
        x_unshuffle = F.pixel_unshuffle(x, downscale_factor=4)

        # reduce channels
        x_unshuffle = self.reduce_channels[2](x_unshuffle)
        x_2_unshuffle = self.reduce_channels[1](x_2_unshuffle)
        x_4 = self.reduce_channels[0](x_4)

        multi_x = torch.cat([x_unshuffle,x_2_unshuffle,x_4], dim=1)

        _out = self.vssm(multi_x)

        _out = resize(_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        _out_ = self.proj_out(_out)
        c2 = c2 + _out_
        c3 = c3 + _out_
        c4 = c4 + _out_

        out += [_out, c2,c3,c4]

        # [batch,1584,40,40]
        out = torch.cat(out, dim=1)

        out = self.cat(out)

        return out

 
    def _forward(self, features):
        x = [features["hl"][i] for i in self.in_index]

        x, c2, c3, c4 = self.forward_mlp_decoder(x)

        feats = self.forward_winssm(x, c2, c3, c4)

        if self.with_ll:
            assert "ll" in features, "Low-level features are required for this model"
            # [b,1024,40,40] -> [b,256,80,80]
            feats = F.pixel_shuffle(feats, upscale_factor=2)
            feats = self.out_conv1(feats)
            # [b,256,80,80] -> [b,64,160,160]
            feats = F.pixel_shuffle(feats, upscale_factor=2)
            feats = self.out_conv2(feats)
            feats_ll = features["ll"].clone()
            feats = self.ll_fusion(feats, feats_ll)

        uncertainty = (
            self.linear_pred_uncertainty(feats).squeeze(1) if self.predict_uncertainty else None
        )

        return feats, uncertainty


    def loss(self, pred, data):
        raise NotImplementedError