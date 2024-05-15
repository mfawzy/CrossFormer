import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """
    def __init__(self, inplanes, planes, use_scale=False, groups=1):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv1d(planes, inplanes, kernel_size=3, padding=1, stride=1,
                                                  groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

    def kernel(self, t, p, g, b, c, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        # t = t.view(b, 1,c//4, 4  * w)
        # p = p.view(b, 1, c//4, 4  * w)
        # g = g.view(b, c//4, 4 * w, 1)

        # t = t.view(b, c//4, 4  * w)
        # p = p.view(b, c//4, 4  * w)
        # g = g.view(b, 4 * w, c//4)

        t = t.view(b, 1,c * w)
        p = p.view(b, 1, c  * w)
        g = g.view(b, c * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c*w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c,  w)
        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c,  w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0., kernel_size=5, dim=17):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv1d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=1)
        self.act = act_layer()
        # self.bn = nn.SyncBatchNorm(in_features)
        self.gn = nn.GroupNorm(num_groups=out_features, num_channels=in_features)
        self.bn = nn.BatchNorm1d(in_features)
        self.conv2 = torch.nn.Conv1d(in_features,  out_features, kernel_size=kernel_size,
                                     padding=padding, groups=1)

        self.conv3 = torch.nn.Conv1d(dim,  dim, kernel_size=kernel_size,
                                     padding=padding, groups=1)
        self.gn1 = nn.GroupNorm(num_groups=out_features, num_channels=out_features)
        # self.gn2 = nn.GroupNorm(num_groups=out_features, num_channels=out_features)
        # self.conv3 = torch.nn.Conv1d(2*in_features, out_features, kernel_size=kernel_size,
        #                              padding=padding, groups=1)

    def forward(self, x):
        res = x
        B, N, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.gn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        x += res
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dim_conv=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.causal = TemporalModelOptimized1f(17, dim, 17, 1)
        if dim_conv == 81:
            self.local = SpatialCGNL(dim_conv, int(dim_conv), use_scale=False, groups=3)
        else:
            self.local_mp = LPI(in_features=dim, act_layer=act_layer, dim=dim_conv)


        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(81)

        eta=1e-5
        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma4 = nn.Parameter(eta * torch.ones(81), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))

        if x.shape[2] == 544:
            # x = x.transpose(-2,-1)
            x =    x + self.drop_path(self.gamma3 * self.local(self.norm3(x)))
            # x = x.transpose(-2,-1)
        else:
            x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x)))

        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x



    # def forward(self, x):
    #     x = x + self.drop_path(self.attn(self.norm1(x)))
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     return x


class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True) # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

attns = []
class PoseTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_patch_to_embedding1 = nn.Linear(in_chans, embed_dim_ratio//2)

        self.temporal_patch_to_embedding1 = nn.Linear(544, 544//4)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed1 = nn.Parameter(torch.zeros(1, 544))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, 544))

        self.chennels_pos_embed = nn.Parameter(torch.zeros(32, 17))


        self.top_pos_embed = nn.Parameter(torch.zeros(1, 544))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        outer_dim=32
        inner_dim=16
        # depth=12
        outer_num_heads=8
        inner_num_heads=4
        mlp_ratio=4.
        qkv_bias=False
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        drop_path_rate=0.
        norm_layer=nn.LayerNorm
        inner_stride=4
        se=1

        self.proj_norm1 = norm_layer(inner_dim)
        self.proj = nn.Linear(inner_dim, outer_dim)
        self.proj_norm2 = norm_layer(outer_dim)

        self.proj_norm12 = norm_layer(136)
        self.proj12 = nn.Linear(136, 544)
        self.proj_norm13 = norm_layer(544)

        self.outer_tokens = nn.Parameter(torch.zeros(1, 17, outer_dim), requires_grad=False)
        self.outer_pos = nn.Parameter(torch.zeros(1, 17, outer_dim))

        self.outer_temp = nn.Parameter(torch.zeros(1, 1, 544))
        self.inner_pos = nn.Parameter(torch.zeros(1, 17, inner_dim))

        self.inner_temp = nn.Parameter(torch.zeros(1, 1, 136))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, dim_conv=17)
            for i in range(depth)])


        self.blocks1 = nn.ModuleList([
            Block(
                dim=544, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, dim_conv=81)
            for i in range(depth)])

        self.lins = nn.Linear(2,32)
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(544)
        self.channels_norm = norm_layer(17)

        self.Temporal = norm_layer(352)
        self.channels = norm_layer(11)

        self.Temporal_norm1 = norm_layer(544)
        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)
        self.weighted_mean1 = torch.nn.Conv1d(in_channels=17, out_channels=17, kernel_size=1)
        self.weighted_mean2 = torch.nn.Conv1d(in_channels=11, out_channels=11, kernel_size=1)




        self.spatial_embed = nn.Parameter(torch.zeros(1, 11, embed_dim_ratio))

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_patch_to_embedding1 = nn.Linear(in_chans, embed_dim_ratio//2)

        self.temporal_patch_to_embedding1 = nn.Linear(544, 544//4)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed1 = nn.Parameter(torch.zeros(1, 544))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, 544))
        self.Temporal_embed = nn.Parameter(torch.zeros(1, 352))


        self.chennels_pos_embed = nn.Parameter(torch.zeros(32, 17))

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )



    def forward_features1(self, x):
        b  = x.shape[0]
        x += self.Temporal_pos_embed1
        x = self.pos_drop(x)
        attn = None
        for blk in self.blocks1:
            x = blk(x)
            attns.append(attn)

        x = self.Temporal_norm1(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x


    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)

        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        b  = x.shape[0]
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean1(x)
        x = x.view(b, 1, -1)
        return x


    def forward(self, x):
        attns.clear()
        x1 = x
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape


        x[:,:,10:16]=0

        #b,17, 32
        x = self.Spatial_forward_features(x)

        #b, 1, 544
        x2=x
        x = self.forward_features1(x)


        x = self.head(x)

        x = x.view(b, 1, p, -1)
        return x



