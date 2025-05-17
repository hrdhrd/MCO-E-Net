# -*- coding: utf-8 -*
import torch.nn as nn
import torch
from opts import parse_opts
from ResNet import ResNet18FeatureExtractor
import torch.nn.functional as F
from MCOMamba import CrossMamba
torch.autograd.set_detect_anomaly(True)

opts = parse_opts()
device = 'cuda:0'

num_experts = 3
top_k = 2
num_tasks = 2


class MCIB(nn.Module):
    def __init__(self, d_model=4):
        super().__init__()
        self.proj_rgb = nn.Linear(d_model, d_model)
        self.proj_event = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=2)
        
    def forward(self, rgb, event):
        """
        rgb: [batch, seq_len, d_model]
        event: [batch, seq_len, d_model]
        """
        rgb = self.proj_rgb(rgb)
        event = self.proj_event(event)
        event_attn, _ = self.attn(
            query=rgb.transpose(0,1), 
            key=event.transpose(0,1), 
            value=event.transpose(0,1)
        ) 
        event_attn = event_attn.transpose(0,1)
        rgb_attn, _ = self.attn(
            query=event.transpose(0,1), 
            key=rgb.transpose(0,1), 
            value=rgb.transpose(0,1)
        )
        rgb_attn = rgb_attn.transpose(0,1)
        gate = torch.sigmoid(torch.sum(rgb * event, dim=-1, keepdim=True))
        fused = gate * event_attn + (1 - gate) * rgb_attn
        return fused + (rgb + event) * 0.5

class AttentionExpert(nn.Module):
    def __init__(self, embed_dim, num_heads, out_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, out_dim)   
    def forward(self, x):
        identity = x
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out.squeeze(0)
        return self.fc(self.norm(attn_out + identity))

class DeepExpert(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim*4),
                nn.Dropout(0.5), 
                nn.Linear(in_dim*4, in_dim)
            ) for _ in range(5)
        ])
        self.final_fc = nn.Linear(in_dim, out_dim)
        self.res_coef = nn.Parameter(torch.tensor(0.3))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + self.res_coef*x
        return self.final_fc(x)

class RouterWithAttention(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, 4)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_experts)
        )
        self.norm = nn.LayerNorm(input_dim)
    def forward(self, x):
        x = x.unsqueeze(0)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.mlp(x.squeeze(0))


class HCEMoE(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1000, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim
        self.experts = nn.ModuleDict({
            f'expert_{i}': self._build_expert(input_dim, output_dim, i%3)
            for i in range(num_experts)
        })
        self.router = RouterWithAttention(input_dim, num_experts)
    def _build_expert(self, in_dim, out_dim, expert_type):
        if expert_type == 0: 
            return nn.Sequential(
                nn.Conv1d(in_dim, 512, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(512, out_dim, kernel_size=1)
            )
        elif expert_type == 1: 
            return AttentionExpert(in_dim, 2, out_dim)
        else:  
            return DeepExpert(in_dim, out_dim)
    def forward(self, x):
        batch_size = x.size(0)
        router_logits = self.router(x)
        expert_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(expert_weights, self.top_k, dim=1)
        final_output = torch.zeros(batch_size, self.output_dim).to(x.device)
        expert_counter = torch.zeros_like(final_output)
        for i in range(self.num_experts):
            expert_mask = (topk_indices == i).any(dim=1)
            if not expert_mask.any():
                continue
            x_masked = x[expert_mask]
            current_expert = self.experts[f'expert_{i}']
            if isinstance(current_expert, nn.Sequential) and isinstance(current_expert[0], nn.Conv1d):
                expert_out = current_expert(x_masked.unsqueeze(-1)).squeeze(-1)
            else:
                expert_out = current_expert(x_masked)
            weights_mask = (topk_indices[expert_mask] == i)
            weights = topk_weights[expert_mask][weights_mask].view(-1, 1)
            if weights.numel() > 0:
                final_output[expert_mask] += expert_out * weights
                expert_counter[expert_mask] += weights
        final_output = final_output / (expert_counter + 1e-8)
        return final_output


class MCOE(nn.Module):
    def __init__(self):
        super(MCOE, self).__init__()
        self.res00 = ResNet18FeatureExtractor()
        self.res01 = ResNet18FeatureExtractor()
        self.res10 = ResNet18FeatureExtractor()
        self.res11 = ResNet18FeatureExtractor()
        self.MCO = CrossMamba(4)
        self.MCIB = MCIB()

        self.moe = HCEMoE(
            input_dim=1024,
            output_dim=7,
            num_experts=8,
            top_k=2
        )
        self.dropout = nn.Dropout(p=0.7)
        self.glu = nn.GLU(dim=-1)
        self.fusion_proj = nn.Sequential(
            nn.Linear(500 * 4, 1024),
            nn.GELU(),
            nn.Dropout(0.5) 
        )
        self.output = nn.Linear(1000, 7)

    def forward(self, frame, event):
        data1F = frame[:, :, 0, :, :]
        data4F = frame[:, :, -1, :, :]
        data1E = event[:, :, 0, :, :]
        data4E = event[:, :, -1, :, :]
        data1_F = self.res00(data1F)
        data4_F = self.res01(data4F)
        data1_E = self.res10(data1E)
        data4_E = self.res11(data4E)
        x_F = torch.cat([data1_F, data4_F], dim=1)
        x_E = torch.cat([data1_E, data4_E], dim=1)
        batch_size = x_E.size(0)
        x_E = x_E.view(batch_size, 500, 4)
        x_F = x_F.view(batch_size, 500, 4)
        rgb_f, event_f = self.MCO(x_F, x_E)
        rgb_f = rgb_f + x_F
        event_f = event_f + x_E
        fused = self.MCIB(rgb_f, event_f).flatten(1)
        chaos_feat = self.fusion_proj(fused)
        output = self.moe(chaos_feat)
        return output

def generate_model():
    model = MCOE()
    return model

def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model
if __name__ == '__main__':
    event_inputs = torch.rand((32, 3, 8, 180, 180)).cuda()
    frame_inputs = torch.rand((32, 3, 8, 180, 180)).cuda()
    Net = MCOE().cuda()
    output = Net(event_inputs, frame_inputs)
