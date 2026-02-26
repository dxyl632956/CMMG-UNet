import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrUpBlock
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch import Tensor

# ==============================================================================

# ==============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep: random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x): return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)  
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].type_as(x)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    def forward(self, x):
        if self.data_format == "channels_last":

            return F.layer_norm(x.type_as(self.weight), self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":

            x_clamp = torch.clamp(x, min=-65504, max=65504) 
            u = x_clamp.mean(1, keepdim=True)
            s = (x_clamp - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x.type_as(self.weight) + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None: x = self.gamma * x
        x = x.permute(0, 3, 1, 2) 
        x = input + self.drop_path(x)
        return x

# ==============================================================================
# 2. GuideDecoder
# ==============================================================================

# class GuideDecoderLayer(nn.Module):
#     def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=24, embed_dim:int=768):
#         super(GuideDecoderLayer, self).__init__()
#         self.text_project = nn.Sequential(
#             nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
#             nn.GELU(), nn.Linear(embed_dim,in_channels), nn.LeakyReLU(),
#         )
#         self.vis_pos = PositionalEncoding(in_channels)
#         self.txt_pos = PositionalEncoding(in_channels,max_len=output_text_len)
#         self.norm1 = nn.LayerNorm(in_channels)
#         self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
#         self.self_attn_norm = nn.LayerNorm(in_channels)
#         self.norm2 = nn.LayerNorm(in_channels)
#         self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
#         self.cross_attn_norm = nn.LayerNorm(in_channels)
#         self.scale = nn.Parameter(torch.tensor(0.01),requires_grad=True)

#     def forward(self,x,txt):
#         txt = self.text_project(txt)
#         # Pre-Norm for Stability
#         vis_norm = self.norm1(x)
#         q = k = self.vis_pos(vis_norm)
#         vis2 = self.self_attn(q, k, value=vis_norm)[0]
#         vis = x + self.self_attn_norm(vis2) # Add & Norm
        
#         vis_norm2 = self.norm2(vis)
#         vis2,_ = self.cross_attn(query=self.vis_pos(vis_norm2), key=self.txt_pos(txt), value=txt)
#         vis = vis + self.scale * self.cross_attn_norm(vis2)
#         return vis

# class GuideDecoder(nn.Module):
#     def __init__(self,in_channels, out_channels, spatial_size, text_len) -> None:
#         super().__init__()
#         self.guide_layer = GuideDecoderLayer(in_channels,text_len)   
#         self.spatial_size = spatial_size
#         self.decoder = UnetrUpBlock(2,in_channels,out_channels,3,2,norm_name='BATCH')
#     def forward(self, vis, skip_vis, txt):
#         if txt is not None: vis = self.guide_layer(vis, txt)
#         vis = rearrange(vis,'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size)
#         skip_vis = rearrange(skip_vis,'B (H W) C -> B C H W',H=self.spatial_size*2,W=self.spatial_size*2)
#         output = self.decoder(vis,skip_vis)
#         output = rearrange(output,'B C H W -> B (H W) C')
#         return output

# ==============================================================================
# 3. Bridger (Swin + Global Attn)
# ==============================================================================

def window_partition(x, window_size: int):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x

class Bridger(nn.Module):
    def __init__(self, d_img, d_model=128, d_txt=768, nhead=8, window_size=7, dropout=0.2, stage_id=1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.stage_id = stage_id
        
        self.proj_v = nn.Conv2d(d_img, d_model, kernel_size=1)
        self.proj_v_out = nn.Conv2d(d_model, d_img, kernel_size=1)
        self.proj_t = nn.Linear(d_txt, d_model)

        self.window_attn = nn.MultiheadAttention(d_model, nhead // 2, dropout=dropout, batch_first=True)
        self.norm_window = nn.LayerNorm(d_model) 
        
        self.fe_channel = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, 1), nn.GELU(),
            nn.Conv2d(d_model // 4, d_model, 1), nn.BatchNorm2d(d_model)
        )
        self.fe_spatial = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, 3, padding=1), nn.GELU(),
            nn.Conv2d(d_model // 4, d_model, 3, padding=1), nn.BatchNorm2d(d_model)
        )
        self.fe_weight = nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.Sigmoid())

        self.vis_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_v_sa = nn.LayerNorm(d_model)
        self.vis_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_v_ca = nn.LayerNorm(d_model)
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.norm_v_ffn = nn.LayerNorm(d_model)
        self.vis_conf_gate = nn.Linear(d_model, 1)
        self.sr = nn.AvgPool2d(kernel_size=2, stride=2) if stage_id <= 2 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.proj_v_out.weight, 0)
        nn.init.constant_(self.proj_v_out.bias, 0)
        nn.init.constant_(self.vis_conf_gate.bias, -3.0) 

    def forward(self, v_in, t_feat_in):

        with torch.cuda.amp.autocast(enabled=False):

            v_in = v_in.float()
            t_feat_in = t_feat_in.float()
            
            
            B, C, H, W = v_in.shape

            v_orig = self.proj_v.float()(v_in)
            t_feat = self.proj_t.float()(t_feat_in)

            # 1. Swin Window Attention
            pad_h, pad_w = (self.window_size - H % self.window_size) % self.window_size, (self.window_size - W % self.window_size) % self.window_size
            v_padded = F.pad(v_orig, (0, pad_w, 0, pad_h))
            Hp, Wp = v_padded.shape[2], v_padded.shape[3]
            v_win = window_partition(v_padded, self.window_size)
            
            v_win_norm = self.norm_window.float()(v_win)
            v_win_attn, _ = self.window_attn.float()(v_win_norm, v_win_norm, v_win_norm)
            v_rec = window_reverse(v_win + self.dropout(v_win_attn), self.window_size, Hp, Wp)[:, :, :H, :W]
            
            # 2. FE Module
            xy = v_orig + v_rec
            v_enhanced = self.fe_weight.float()(xy) * self.fe_channel.float()(xy) + \
                         (1.0 - self.fe_weight.float()(xy)) * self.fe_spatial.float()(xy)

            # 3. Global Self-Attention
            v_flat = rearrange(v_enhanced, 'b c h w -> b (h w) c')
            v_kv = rearrange(self.sr.float()(v_enhanced), 'b c h w -> b (h w) c')
            
            v_flat_norm = self.norm_v_sa.float()(v_flat)
            v_kv_norm = self.norm_v_sa.float()(v_kv)
            v_sa, _ = self.vis_self_attn.float()(v_flat_norm, v_kv_norm, v_kv_norm)
            v_flat = v_flat + self.dropout(v_sa)

            # 4. Cross-Attention
            v_flat_norm = self.norm_v_ca.float()(v_flat)
            v_ca, _ = self.vis_cross_attn.float()(query=v_flat_norm, key=t_feat, value=t_feat)
            v_flat = v_flat + self.dropout(v_ca)

            # 5. FFN
            v_flat_norm = self.norm_v_ffn.float()(v_flat)
            v_ffn_out = self.vis_ffn.float()(v_flat_norm)
            v_gate = torch.sigmoid(self.vis_conf_gate.float()(v_ffn_out))
            v_flat = v_flat + v_gate * self.dropout(v_ffn_out)

            p_vis = rearrange(v_flat, 'b (h w) c -> b c h w', h=H, w=W)
            out = self.proj_v_out.float()(p_vis)
            
            return out.type_as(v_in)
# ==============================================================================
# 4. Decoder Attention Modules
# ==============================================================================

class ECA_ChannelAttention(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECA_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x0 = x
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x0*y.expand_as(x0)

class Dec_ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Dec_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x0 = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x0*self.sigmoid(out)

class Dec_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Dec_SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x*self.sigmoid(out)

# ==============================================================================
# 6. MK-UNet(Multi-kernel Lightweight Components)
# ==============================================================================

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6'):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, 
                          kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU6(inplace=True)
            )
            for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        out = 0
        for dwconv in self.dwconvs:
            out += dwconv(x)
        return out

class MultiKernelInvertedResidualBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1, expansion_factor=2, kernel_sizes=[1,3,5]):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.use_skip_connection = (self.stride == 1 and in_c == out_c)


        self.ex_c = int(self.in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.ex_c),
            nn.ReLU6(inplace=True)
        )        
        

        self.mk_dwconv = MultiKernelDepthwiseConv(self.ex_c, kernel_sizes, stride)


        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.ex_c, self.out_c, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_c),
        )

    def forward(self, x):
        identity = x
        
        x = self.pconv1(x)
        x = self.mk_dwconv(x)
        

        x = channel_shuffle(x, groups=self.in_c // 2 if self.in_c > 1 else 1) 
        
        x = self.pconv2(x)

        if self.use_skip_connection:
            return identity + x
        else:
            return x

class GroupedAttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1):
        super(GroupedAttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)
            
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class MK_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(MK_SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))

class MK_ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(MK_ChannelAttention, self).__init__()

        if in_planes < ratio: ratio = in_planes 
        reduced = in_planes // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, reduced, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)