import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import time

# --- 核心上采样启动函数 (保持不变) ---

def UPA(HR_img, lr_modality):
    with torch.enable_grad():
        start_time = time.time()
        USE_AMP = True
        AMP_DTYPE = torch.float16
        
        hr = torch.from_numpy(np.array(HR_img)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0  
        H, W = hr.shape[-2:]
        Hl, Wl = lr_modality.shape[-2:]
        scale = int(H / Hl)
        
        lr_train_input = F.interpolate(hr, scale_factor=1/scale, mode="bicubic", align_corners=False)
        model = LearnablePixelwiseAnisoJBU_NoParent(Hl, Wl, scale=scale).cuda()
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=1e-1)
        max_steps = 20 # 对应下方循环步数
        gamma = (1e-9 / 1e-1) ** (1.0 / 5100)
        scheduler = LambdaLR(opt, lr_lambda=lambda step: gamma ** step)
        scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

        print(f"\n[UPA-Fast] 启动优化 | 尺寸: {H}x{W} | 缩放: {scale}x")

        for step in range(1, 21):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
                pred = model(lr_train_input, hr) 
                loss = F.l1_loss(pred, hr)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            if step % 10 == 0 or step == 1:
                mem = torch.cuda.memory_allocated() / 1024**2
                print(f"  > Step {step:2d}/20 | Loss: {loss.item():.6f} | Mem: {mem:.1f}MB")

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            hr_feat = model(lr_modality.to(torch.float32), hr)
        
        print(f"[UPA-Fast] 完成 | 总耗时: {time.time() - start_time:.2f}s")
        return hr_feat

# --- 核心算子实现 ---

def _tanh_bound_pi(raw):
    return math.pi * torch.tanh(raw)

def gs_jbu_aniso_noparent(feat_lr, guide_hr, scale, sx_map, sy_map, th_map, sr_map, R_max=4, alpha_dyn=2.0):
    B, C, Hl, Wl = feat_lr.shape
    _, _, Hh, Wh = guide_hr.shape
    dev = feat_lr.device

    # 1. 坐标映射：计算 HR 像素对应 LR 的中心位置
    y_hr = torch.arange(Hh, device=dev).float()
    x_hr = torch.arange(Wh, device=dev).float()
    Yh, Xh = torch.meshgrid(y_hr, x_hr, indexing='ij')

    u_lr = (Yh + 0.5) / scale - 0.5
    v_lr = (Xh + 0.5) / scale - 0.5
    uc = torch.round(u_lr).long()
    vc = torch.round(v_lr).long()

    # 2. 参数上采样至 HR 空间 (Nearest 保证像素对齐)
    def up(p): return F.interpolate(p, (Hh, Wh), mode='nearest').squeeze(0).squeeze(0)
    sx = up(sx_map).clamp_min(1e-6)
    sy = up(sy_map).clamp_min(1e-6)
    th = up(th_map)
    sr = up(sr_map).clamp_min(1e-6)
    
    # 动态半径掩码准备
    R_map_sq = (alpha_dyn * torch.maximum(sx, sy)).clamp(1, R_max)**2

    # 3. 准备引导图
    guide_lr = F.interpolate(guide_hr, size=(Hl, Wl), mode='bilinear', align_corners=False)
    
    # 初始化累加器
    num = torch.zeros((C, Hh, Wh), device=dev, dtype=torch.float32)
    den = torch.zeros((Hh, Wh), device=dev, dtype=torch.float32)
    
    # 预计算旋转三角函数
    cos_t, sin_t = torch.cos(th), torch.sin(th)

    # 4. 遍历邻域偏移量 (取代原代码的像素级循环)
    # R_max=8 时循环 289 次，比原代码几百万次循环快得多
    for dy_off in range(-R_max, R_max + 1):
        for dx_off in range(-R_max, R_max + 1):
            # 计算当前偏移在 LR 空间的索引
            ui = (uc + dy_off).clamp(0, Hl - 1)
            vi = (vc + dx_off).clamp(0, Wl - 1)
            
            # 计算当前偏移在 HR 空间的实际距离 dx, dy
            # dx = 当前HR像素x - 对应LR采样点中心x
            cur_dx = (Xh - (vi.float() * scale + (scale - 1) / 2.0)) / scale
            cur_dy = (Yh - (ui.float() * scale + (scale - 1) / 2.0)) / scale
            
            # 空间权重 (Anisotropic Spatial)
            log_w = -((cur_dx * cos_t + cur_dy * sin_t)**2) / (2 * sx**2 + 1e-8) \
                    -((-cur_dx * sin_t + cur_dy * cos_t)**2) / (2 * sy**2 + 1e-8)
            
            # 值域权重 (Range)
            # 使用高级索引一次性提取所有像素对应的引导值
            g_lr_sampled = guide_lr[0, :, ui, vi] # [3, Hh, Wh]
            g_diff2 = ((guide_hr[0] - g_lr_sampled)**2).sum(0)
            log_w += -g_diff2 / (2.0 * sr**2 + 1e-8)
            
            # 半径掩码
            mask = (dy_off**2 + dx_off**2) <= R_map_sq
            
            # 计算权重并累加
            w = torch.exp(log_w) * mask.float()
            
            # 采样特征并累加
            f_lr_sampled = feat_lr[0, :, ui, vi] # [C, Hh, Wh]
            num += f_lr_sampled * w.unsqueeze(0)
            den += w

    res = num / den.clamp_min(1e-8)
    return res.unsqueeze(0).to(feat_lr.dtype)

# --- 模型类 (保持不变) ---

class LearnablePixelwiseAnisoJBU_NoParent(nn.Module):
    def __init__(self, Hl, Wl, scale=16, init_sigma=16.0, init_sigma_r=0.12, R_max=8, alpha_dyn=2.0):
        super().__init__()
        self.scale, self.R_max, self.alpha_dyn = scale, R_max, alpha_dyn
        self.sx_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.sy_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.th_raw = nn.Parameter(torch.zeros((1, 1, Hl, Wl)))
        self.sr_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma_r))))

    def forward(self, feat_lr, guide_hr):
        return gs_jbu_aniso_noparent(
            feat_lr, guide_hr, self.scale, 
            torch.exp(self.sx_raw), torch.exp(self.sy_raw), 
            _tanh_bound_pi(self.th_raw), torch.exp(self.sr_raw), 
            R_max=self.R_max, alpha_dyn=self.alpha_dyn
        )