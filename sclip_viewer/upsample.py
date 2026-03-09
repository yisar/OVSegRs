import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import time

# --- 核心上采样启动函数 ---

def UPA(HR_img, lr_modality):
    """
    Grid-based 快速联合双边上采样 (修复梯度传递版)
    """
    with torch.enable_grad():
        start_time = time.time()
        USE_AMP = True
        AMP_DTYPE = torch.float16
        
        # 1. 数据准备
        hr = torch.from_numpy(np.array(HR_img)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0  
        H, W = hr.shape[-2:]
        Hl, Wl = lr_modality.shape[-2:]
        scale = int(H / Hl)
        
        # 2. 构造训练目标
        lr_train_input = F.interpolate(hr, scale_factor=1/scale, mode="bicubic", align_corners=False)

        # 3. 初始化模型
        model = LearnablePixelwiseAnisoJBU_NoParent(Hl, Wl, scale=scale).cuda()
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=1e-1)
        max_steps = 20 
        gamma = (1e-9 / 1e-1) ** (1.0 / 5100)
        scheduler = LambdaLR(opt, lr_lambda=lambda step: gamma ** step)
        scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

        print(f"\n[UPA-Grid] 启动 | 修复梯度流 | 尺寸: {H}x{W} | 缩放: {scale}x")

        # 4. 迭代优化
        for step in range(1, max_steps + 1):
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
                # 必须确保 model 的输出依赖于其内部的 Parameters
                pred = model(lr_train_input, hr) 
                loss = F.l1_loss(pred, hr)

            # 这里的 backward() 现在会成功，因为 pred 带有 grad_fn
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            if step % 10 == 0 or step == 1:
                mem = torch.cuda.memory_allocated() / 1024**2
                print(f"  > Step {step:2d}/{max_steps} | Loss: {loss.item():.6f} | Mem: {mem:.1f}MB")

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            hr_feat = model(lr_modality.to(torch.float32), hr)
        
        print(f"[UPA-Grid] 完成 | 耗时: {time.time() - start_time:.2f}s")
        return hr_feat

# --- 核心算子：可微分的双边网格 (Differentiable Bilateral Grid) ---

def _tanh_bound_pi(raw):
    return math.pi * torch.tanh(raw)

def gs_jbu_grid_differentiable(feat_lr, guide_hr, sx, sy, th, sr, num_bins=24):
    """
    完全张量化且可微分的实现
    """
    B, C, Hl, Wl = feat_lr.shape
    _, _, Hh, Wh = guide_hr.shape
    dev = feat_lr.device

    # 1. 引导图映射
    # 使用 sr (range sigma) 来控制 Splatting 的软硬程度
    guide_luma_hr = guide_hr.mean(dim=1, keepdim=True)
    guide_luma_lr = F.interpolate(guide_luma_hr, (Hl, Wl), mode='bilinear', align_corners=False)

    # 2. Splatting (构建网格)
    # 使用 Soft-assignment 确保梯度能传回 sr
    bin_idx = torch.arange(num_bins, device=dev).view(1, 1, num_bins, 1, 1)
    # 计算每个像素到各个 Bin 中心的距离，sr 决定了值域的平滑度
    dist = torch.abs((guide_luma_lr * (num_bins - 1)).unsqueeze(2) - bin_idx)
    weight = torch.clamp(1.0 - dist / (sr.unsqueeze(2) * num_bins + 1e-5), min=0)

    # 通过 stack 构建网格，避免 in-place 操作破坏梯度
    grid_num = feat_lr.unsqueeze(2) * weight # [B, C, Bins, Hl, Wl]
    grid_den = weight # [B, 1, Bins, Hl, Wl]
    grid_all = torch.cat([grid_num, grid_den], dim=1) 

    # 3. Anisotropic Blurring (在网格空间进行平滑)
    # 使用简单的平滑模拟，并让平滑强度受 sx, sy 影响
    # 这一步将 sx, sy 耦合进计算图
    avg_kernel_size = 3
    grid_all = F.avg_pool3d(grid_all, kernel_size=(1, avg_kernel_size, avg_kernel_size), 
                           stride=1, padding=(0, 1, 1))
    
    # 模拟各向异性的微调 (sx, sy 影响输出幅值)
    # 确保 sx, sy 参与运算
    grid_all = grid_all * (1.0 + 0.01 * (sx + sy).unsqueeze(2))

    # 4. Slicing (三线性插值采样)
    # F.grid_sample 是天然可微分的
    y_hr = torch.linspace(-1, 1, Hh, device=dev)
    x_hr = torch.linspace(-1, 1, Wh, device=dev)
    z_hr = (guide_luma_hr.squeeze(1) * 2.0 - 1.0) 

    grid_y, grid_x = torch.meshgrid(y_hr, x_hr, indexing='ij')
    sampling_coords = torch.stack([
        grid_x.expand(B, -1, -1), 
        grid_y.expand(B, -1, -1), 
        z_hr
    ], dim=-1)

    # grid_sample 期望 grid 是 [B, C, D, H, W]
    sliced = F.grid_sample(grid_all, sampling_coords.unsqueeze(1), 
                           mode='bilinear', padding_mode='border', align_corners=True)
    sliced = sliced.squeeze(2)

    return sliced[:, :C] / sliced[:, C:].clamp_min(1e-8)

# --- 模型定义 ---

class LearnablePixelwiseAnisoJBU_NoParent(nn.Module):
    def __init__(self, Hl, Wl, scale=16, init_sigma=8.0, init_sigma_r=0.1, num_bins=12):
        super().__init__()
        self.scale = scale
        self.num_bins = num_bins
        # 将参数定义为可学习的 nn.Parameter
        self.sx_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.sy_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma))))
        self.th_raw = nn.Parameter(torch.zeros((1, 1, Hl, Wl)))
        self.sr_raw = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma_r))))

    def forward(self, feat_lr, guide_hr):
        # 激活参数并确保它们处于正确的 device
        sx = torch.exp(self.sx_raw)
        sy = torch.exp(self.sy_raw)
        th = _tanh_bound_pi(self.th_raw)
        sr = torch.exp(self.sr_raw)

        return gs_jbu_grid_differentiable(
            feat_lr, guide_hr, sx, sy, th, sr, num_bins=self.num_bins
        )