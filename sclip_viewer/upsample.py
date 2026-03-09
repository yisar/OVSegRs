import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import time

# --- 核心工具函数 ---
@torch.no_grad()
def _build_offsets(R_max, device):
    """预生成邻域偏移量"""
    offs = torch.arange(-R_max, R_max + 1, device=device)
    dY, dX = torch.meshgrid(offs, offs, indexing='ij')
    # 强制形状为 [Nn, 1, 1]
    return dY.reshape(-1, 1, 1), dX.reshape(-1, 1, 1)

def _tanh_bound_pi(raw):
    return math.pi * torch.tanh(raw)

# --- 向量化核心算子 (终极防御版) ---
def gs_jbu_aniso_vectorized(feat_lr, guide_hr, scale, sx_map, sy_map, th_map, sr_map, 
                            R_max=4, alpha_dyn=2.0):
    B, C, Hl, Wl = feat_lr.shape
    _, Ch, Hh, Wh = guide_hr.shape
    dev = feat_lr.device
    dtype_feat = feat_lr.dtype

    # 1. 生成精准坐标
    y_hr = torch.arange(Hh, device=dev).float().view(1, Hh, 1).expand(1, Hh, Wh)
    x_hr = torch.arange(Wh, device=dev).float().view(1, 1, Wh).expand(1, Hh, Wh)
    
    uc = torch.round((y_hr + 0.5) / scale - 0.5).long().clamp(0, Hl-1)
    vc = torch.round((x_hr + 0.5) / scale - 0.5).long().clamp(0, Wl-1)

    # 2. 获取邻域偏移与索引
    dY, dX = _build_offsets(R_max, dev)
    Nn = dY.shape[0]  # R_max=4 时，Nn=81
    
    Ui = (uc + dY).clamp(0, Hl-1) # [Nn, Hh, Wh]
    Vi = (vc + dX).clamp(0, Wl-1) # [Nn, Hh, Wh]
    idx_flat = (Ui * Wl + Vi).reshape(-1)

    # 3. 极速采样器
    def fast_sample(src_map, channels):
        src_flat = src_map.view(channels, -1)
        sampled = src_flat.index_select(1, idx_flat)
        return sampled.view(channels, Nn, Hh, Wh)

    # 采样参数图并直接剥离多余的通道维，保证输出严格为 [Nn, Hh, Wh]
    sx = fast_sample(sx_map, 1).squeeze(0).clamp_min(1e-6) 
    sy = fast_sample(sy_map, 1).squeeze(0).clamp_min(1e-6)
    th = fast_sample(th_map, 1).squeeze(0)
    sr = fast_sample(sr_map, 1).squeeze(0).clamp_min(1e-6)

    # 4. 计算空间权重 (各向异性)
    dx_vec = x_hr.squeeze(0) - ((Vi.float() + 0.5) * scale - 0.5) # [Nn, Hh, Wh]
    dy_vec = y_hr.squeeze(0) - ((Ui.float() + 0.5) * scale - 0.5) # [Nn, Hh, Wh]
    
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    x_rot = dx_vec * cos_t + dy_vec * sin_t
    y_rot = -dx_vec * sin_t + dy_vec * cos_t
    
    log_w_spatial = -(x_rot**2 / (2 * sx**2 + 1e-8)) - (y_rot**2 / (2 * sy**2 + 1e-8))

    # 5. 计算颜色权重 (Range)
    guide_lr = F.interpolate(guide_hr, (Hl, Wl), mode='bilinear', align_corners=False)
    g_lr_sampled = fast_sample(guide_lr, Ch) # [Ch, Nn, Hh, Wh]
    
    # 严格对齐后求差值并沿通道(Ch)维度求和
    diff = guide_hr[0].unsqueeze(1) - g_lr_sampled # [Ch, 1, Hh, Wh] - [Ch, Nn, Hh, Wh]
    g_diff2 = (diff ** 2).sum(dim=0) # -> [Nn, Hh, Wh]
    log_w_color = -g_diff2 / (2.0 * sr**2 + 1e-8)

    # 6. 权重归一化与掩码
    sigma_eff_lr = torch.maximum(sx_map, sy_map)
    sigma_eff_hr = F.interpolate(sigma_eff_lr, (Hh, Wh), mode='bilinear', align_corners=False)
    R_map = torch.ceil(alpha_dyn * sigma_eff_hr).clamp(1, R_max).squeeze(0).squeeze(0) # [Hh, Wh]
    
    dist_sq = dY**2 + dX**2 # [Nn, 1, 1]
    mask = dist_sq <= (R_map**2).unsqueeze(0) # [Nn, Hh, Wh]
    
    log_w = log_w_spatial + log_w_color
    log_w = torch.where(mask, log_w, torch.full_like(log_w, float("-inf")))
    
    w = F.softmax(log_w, dim=0) # [Nn, Hh, Wh]

    # 7. 特征加权 (采用 Einsum 确保维度绝对正确)
    feat_lr_flat = feat_lr.view(C, -1)
    feat_sampled = feat_lr_flat.index_select(1, idx_flat).view(C, Nn, Hh, Wh)
    
    # 爱因斯坦求和约定：c=通道, n=邻域, h=高度, w=宽度。沿着 n 维度求和
    feat_hr = torch.einsum('cnhw,nhw->chw', feat_sampled, w)
    
    return feat_hr.unsqueeze(0).to(dtype_feat)

# --- 模型定义 ---

class LearnablePixelwiseAnisoJBU_Optimized(nn.Module):
    def __init__(self, Hl, Wl, scale=16, init_sigma=16.0, init_sigma_r=0.12, R_max=4):
        super().__init__()
        self.scale, self.R_max = scale, R_max
        self.sx_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.sy_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.th_raw = nn.Parameter(torch.zeros((1, 1, Hl, Wl)))
        self.sr_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma_r))))

    def forward(self, feat_lr, guide_hr):
        return gs_jbu_aniso_vectorized(
            feat_lr, guide_hr, self.scale, 
            torch.exp(self.sx_raw), torch.exp(self.sy_raw), 
            _tanh_bound_pi(self.th_raw), torch.exp(self.sr_raw), 
            R_max=self.R_max
        )

# --- 业务启动入口 ---

def UPA_Optimized(HR_img, lr_modality):
    with torch.enable_grad():
        start_time = time.time()
        
        # [严密的张量防御机制] 确保 hr 绝对为 [1, 3, H, W]
        if not torch.is_tensor(HR_img):
            hr = torch.from_numpy(np.array(HR_img)).float().cuda() / 255.0
            if hr.ndim == 3: hr = hr.permute(2, 0, 1).unsqueeze(0)
        else:
            hr = HR_img.clone().detach().float().cuda()
            if hr.max() > 2.0: hr = hr / 255.0
            # 修复上游传入 [H, W, 3] 或 [3, H, W] 的异常情况
            if hr.ndim == 3:
                if hr.shape[-1] == 3:  # [H, W, 3] -> [1, 3, H, W]
                    hr = hr.permute(2, 0, 1).unsqueeze(0)
                elif hr.shape[0] == 3: # [3, H, W] -> [1, 3, H, W]
                    hr = hr.unsqueeze(0)
            elif hr.ndim == 4 and hr.shape[-1] == 3: # [1, H, W, 3] -> [1, 3, H, W]
                hr = hr.permute(0, 3, 1, 2)

        Hh, Wh = hr.shape[-2:]
        Hl, Wl = lr_modality.shape[-2:]
        scale = Hh // Hl

        # 构造自监督训练输入
        lr_train_input = F.interpolate(hr, size=(Hl, Wl), mode="bicubic", align_corners=False)
        model = LearnablePixelwiseAnisoJBU_Optimized(Hl, Wl, scale=scale, R_max=4).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        
        for _ in range(10):
            opt.zero_grad(set_to_none=True)
            pred = model(lr_train_input, hr)
            loss = F.l1_loss(pred, hr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            hr_feat = model(lr_modality.float(), hr)

        return hr_feat