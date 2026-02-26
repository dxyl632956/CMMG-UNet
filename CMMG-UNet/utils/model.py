import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import *
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from einops import rearrange
import math

# ==============================================================================

# ==============================================================================
class BERTModel(nn.Module):
    def __init__(self, bert_type, project_dim):
        super(BERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(           
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),           
            nn.GELU(),           
            nn.Linear(project_dim, project_dim)
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) 
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) 
        embed = self.project_head(embed)
        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):
    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True) 
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)
        return {"feature":output['hidden_states'], "project":project}

# ==============================================================================
# 2. Memory & Attention Components
# ==============================================================================
class MemoryAttention(nn.Module):
    def __init__(self, input_dim, mem_dim):
        super(MemoryAttention, self).__init__()
        self.input_dim = input_dim
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(input_dim + mem_dim, input_dim, kernel_size=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU()
        )
        self.offset_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim)
        )
        self.learnable_gamma = nn.Parameter(torch.ones(1) * 0.5)
        self.norm_out = nn.LayerNorm(input_dim)

    def forward(self, query_feat, retrieved_mem_list, retrieval_confidence):
        B, C, H, W = query_feat.shape
        max_sim, _ = torch.max(retrieval_confidence, dim=1)
        quality_mask = (max_sim > 0.35).float().view(B, 1, 1, 1)
        weights = F.softmax(retrieval_confidence, dim=1).view(B, -1, 1, 1, 1)
        mem_prior = (retrieved_mem_list * weights).sum(dim=1) 
        fused = torch.cat([query_feat, mem_prior], dim=1)
        refined = self.fusion_conv(fused)
        offset = self.offset_net(refined.mean([2, 3])).view(B, C, 1, 1)
        avg_sim = retrieval_confidence.mean(dim=1).view(B, 1, 1, 1)
        participation = (0.3 + torch.sigmoid(self.learnable_gamma) * 0.4 + avg_sim * 0.3) * quality_mask
        output = query_feat + participation * (refined + offset)
        output = output.permute(0, 2, 3, 1)
        output = self.norm_out(output)
        return output.permute(0, 3, 1, 2)

class MemoryEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(MemoryEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Tanh() 
        )
    def forward(self, image, mask):
        mask = mask.expand_as(image)
        x = image + mask
        out = self.conv(x)
        return out

# ==============================================================================
# 3. Memory Bank 
# ==============================================================================
class MemoryBank(nn.Module):
    def __init__(self, memory_size_N, text_dim=768, feature_dims=[96, 192, 384, 768], cluster_k=5):
        super(MemoryBank, self).__init__()
        self.N = memory_size_N
        self.cluster_k = cluster_k 
        self.register_buffer('text_bank', torch.zeros(self.N, 24, text_dim, dtype=torch.float16))
        for i in range(4):
            sz = [56, 28, 14, 7][i]
            self.register_buffer(f'visual_bank_{i}', torch.zeros(self.N, feature_dims[i], sz, sz, dtype=torch.float16))
        self.register_buffer('utility_score', torch.zeros(self.N)) 
        self.register_buffer('age', torch.zeros(self.N))          
        self.register_buffer('current_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('is_full', torch.tensor(0, dtype=torch.long))
        self.register_buffer('loss_moving_avg', torch.tensor(0.08)) 

    def get_matches_dual_sim(self, text_query, vision_query_map, k=6):
        with torch.no_grad():
            curr_size = self.current_ptr.item() if self.is_full.item() == 0 else self.N
            if curr_size == 0:
                return [torch.zeros_like(getattr(self, f'visual_bank_{i}')[0:k]) for i in range(4)], torch.zeros(text_query.shape[0], k).to(text_query.device)

            valid_text_bank = self.text_bank[:curr_size]
            valid_vis_bank_3 = self.visual_bank_3[:curr_size]

            q_t = F.normalize(text_query.detach().float().mean(1), p=2, dim=1)
            b_t = F.normalize(valid_text_bank.float().mean(1), p=2, dim=1)
            t_sim = torch.mm(q_t, b_t.t()) 
            
            saliency = torch.sigmoid(vision_query_map.detach().mean(1, keepdim=True))
            q_v = (vision_query_map.detach().float() * saliency).sum([2,3]) / (saliency.sum([2,3]) + 1e-6)
            q_v = F.normalize(q_v, p=2, dim=1)
            b_v = F.normalize(valid_vis_bank_3.float().mean([2,3]), p=2, dim=1)
            v_sim = torch.mm(q_v, b_v.t()) 
            
            total_sim = 0.5 * t_sim + 0.5 * v_sim
            valid_mask = (total_sim > 0.4).float() 
            
            real_k = min(k, curr_size)
            vals, indices = torch.topk(total_sim, k=real_k, dim=1)
            topk_mask = torch.gather(valid_mask, 1, indices)
            vals = vals * topk_mask
            
            self.age += 1
            self.age[indices] = 0
            
        retrieved_vs = [getattr(self, f'visual_bank_{i}')[indices] for i in range(4)]
        if real_k < k:
            pad_size = k - real_k
            retrieved_vs = [F.pad(v, (0,0,0,0,0,0,0,pad_size)) for v in retrieved_vs]
            vals = F.pad(vals, (0, pad_size), value=0.0) 
        return retrieved_vs, vals

    def _run_kmeans(self, x, k):
        N, D = x.shape
        if N < k: return torch.zeros(N).long().to(x.device)
        indices = torch.randperm(N)[:k]
        centroids = x[indices]
        for _ in range(10): 
            dists = torch.cdist(x, centroids)
            labels = torch.argmin(dists, dim=1)
            new_centroids = []
            for i in range(k):
                mask = (labels == i)
                if mask.sum() > 0:
                    new_centroids.append(x[mask].mean(0))
                else:
                    random_idx = torch.randint(0, N, (1,))
                    new_centroids.append(x[random_idx].squeeze(0))
            centroids = torch.stack(new_centroids)
        return labels

    def update_wld(self, loss_val, t_feat, v_feats_list):
        if self.loss_moving_avg.item() < 0: self.loss_moving_avg.fill_(loss_val)
        target_idx = 0
        
        # Phase 1: Fast Fill
        if self.is_full.item() == 0:
            target_idx = self.current_ptr.item()
            self.current_ptr += 1
            if self.current_ptr >= self.N: self.is_full.fill_(1)
        
        # Phase 2: Clustering Eviction
        else:
            if loss_val < self.loss_moving_avg * 0.7: return
            with torch.no_grad():
                t_rep = F.normalize(self.text_bank.mean(1), p=2, dim=1)
                v_rep = F.normalize(self.visual_bank_3.mean([2, 3]), p=2, dim=1)
                feats = 0.5 * t_rep + 0.5 * v_rep
                labels = self._run_kmeans(feats, self.cluster_k)
                counts = torch.bincount(labels, minlength=self.cluster_k)
                largest_cluster_idx = torch.argmax(counts).item()
                candidate_indices = torch.where(labels == largest_cluster_idx)[0]
                idx_in_candidates = torch.randint(0, len(candidate_indices), (1,)).item()
                target_idx = candidate_indices[idx_in_candidates].item()
        
        target_idx = int(max(0, min(target_idx, self.N - 1)))
        self.text_bank[target_idx] = t_feat[0].detach().half()
        for i in range(4):
            getattr(self, f'visual_bank_{i}')[target_idx] = v_feats_list[i][0].detach().half()
        self.loss_moving_avg = self.loss_moving_avg * 0.95 + loss_val * 0.05
        self.utility_score[target_idx] = loss_val
        self.age[target_idx] = 0

    def reset(self):
        self.current_ptr.fill_(0); self.is_full.fill_(0); self.utility_score.fill_(0); self.age.fill_(0)
        self.loss_moving_avg.fill_(0.08)


# ==============================================================================
# 4. MK-UNet Decoder Block (New)
# ==============================================================================
class MKDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(MKDecoderBlock, self).__init__()
        self.ca = MK_ChannelAttention(in_channels)
        self.sa = MK_SpatialAttention()
        self.mkir = MultiKernelInvertedResidualBlock(in_channels, out_channels, stride=1, kernel_sizes=[1,3,5])
        self.gag = GroupedAttentionGate(F_g=out_channels, F_l=skip_channels, F_int=out_channels//2)
        self.skip_align = nn.Identity()
        if skip_channels != out_channels:
            self.skip_align = nn.Conv2d(skip_channels, out_channels, 1, bias=False)

    def forward(self, x, skip):
        x = self.ca(x)
        x = self.sa(x)
        x = self.mkir(x)
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x_up.size()[2:] != skip.size()[2:]:
            x_up = F.interpolate(x_up, size=skip.size()[2:], mode='bilinear', align_corners=True)
        skip_gated = self.gag(g=x_up, x=skip)
        skip_aligned = self.skip_align(skip_gated)
        out = x_up + skip_aligned
        return out

# ==============================================================================
# 5. Main Model
# ==============================================================================
class KMMG_UNet(nn.Module):
    def __init__(self, bert_type, vision_type, project_dim=768, 
                 memory_N=512, memory_K=3, cluster_k=100, warmup_epochs=20):
        super(KMMG_UNet, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.memory_K = memory_K
        self.cluster_k = cluster_k 
        
        # --- Encoder ---
        depths = [3, 3, 9, 3] 
        dims = [96, 192, 384, 768]
        drop_path_rate = 0.3
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.downsample_layers = nn.ModuleList() 
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # --- Memory & Text ---
        self.text_encoder = BERTModel(bert_type, project_dim)
        self.memory_bank = MemoryBank(memory_N, project_dim, dims, cluster_k=self.cluster_k)
        
        self.visual_memory_attns = nn.ModuleList([
            nn.Identity(), nn.Identity(),
            MemoryAttention(dims[2], dims[2]), 
            MemoryAttention(dims[3], dims[3]) 
        ])
        
        self.memory_encoder = MemoryEncoder(3, 3)
        self.fusion1 = Bridger(dims[0], dims[0], stage_id=1)
        self.fusion2 = Bridger(dims[1], dims[1], stage_id=2)
        self.fusion3 = Bridger(dims[2], dims[2], stage_id=3)
        self.fusion4 = Bridger(dims[3], dims[3], stage_id=4)
      

        self.decode4 = MKDecoderBlock(in_channels=dims[3], skip_channels=dims[2], out_channels=dims[2])
        self.decode3 = MKDecoderBlock(in_channels=dims[2], skip_channels=dims[1], out_channels=dims[1])
        self.decode2 = MKDecoderBlock(in_channels=dims[1], skip_channels=dims[0], out_channels=dims[0])
        
  
        self.decoder1 = SubpixelUpsample(2, dims[0], 24, 4)
        self.out = UnetOutBlock(2, 24, 1)

    def _forward_visual_encoder(self, images):
        outs = []
        x = images
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x) 
        return outs

    def forward(self, data, target=None, current_epoch=0):
        image, text = data
        if image.shape[1] == 1: image = repeat(image, 'b 1 h w -> b c h w', c=3)

        # 1. Encode
        V_raw_list = self._forward_visual_encoder(image) 
        T_feat = self.text_encoder(text['input_ids'], text['attention_mask'])['feature'][-1]
        V_star_list = list(V_raw_list) 
        
        # 2. Memory Retrieval
        if (self.memory_bank.is_full == 1 or self.memory_bank.current_ptr > 64) and current_epoch >= self.warmup_epochs:
            retrieved_vs, confidence = self.memory_bank.get_matches_dual_sim(T_feat, V_raw_list[3], k=6)
            for i in [2, 3]:
                V_star_list[i] = self.visual_memory_attns[i](V_raw_list[i], retrieved_vs[i], confidence)

        # 3. Fusion
        P_list = []
        bridgers = [self.fusion1, self.fusion2, self.fusion3, self.fusion4]
        for v_feat, bridger in zip(V_star_list, bridgers):
            P_list.append(bridger(v_feat, T_feat))

        encoder_feats = [v_raw + p_itm for v_raw, p_itm in zip(V_raw_list, P_list)]

        # 4. Decode (使用 MKDecoderBlock)
        d4 = self.decode4(encoder_feats[3], encoder_feats[2])
        d3 = self.decode3(d4, encoder_feats[1])
        d2 = self.decode2(d3, encoder_feats[0])
        
        out_logits = self.out(self.decoder1(d2))

        # 5. Memory Collection
        mem_data = None
        if self.training and target is not None:
            with torch.no_grad():
                refined_collect_list = []
                for i in range(4):
                    curr_v_star = V_star_list[i]
                    curr_v_raw = V_raw_list[i]
                    B_c, C_c, H_c, W_c = curr_v_star.shape
                    gt_mask = F.interpolate(target.float(), size=(H_c, W_c), mode='nearest')
                    v_refined = curr_v_star + 0.5 * (curr_v_raw * gt_mask)
                    refined_collect_list.append(v_refined)
                mem_data = (T_feat.detach(), [f.detach() for f in refined_collect_list])

        return out_logits, mem_data