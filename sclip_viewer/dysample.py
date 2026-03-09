import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySampleTTO(nn.Module):
    """改造后的 TTO 版本 DySample
    核心改动：
    1. 新增 TTO 优化配置（学习率、步数、损失函数）
    2. 新增 test_time_optimize 方法，专用于测试时微调
    3. 支持冻结/解冻模块参数，仅优化 DySample 自身
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False,
                 tto_lr=1e-4, tto_steps=5, tto_loss_type='reconstruction'):
        super().__init__()
        # 原始 DySample 初始化
        self.scale = scale
        self.style = style
        self.groups = groups
        self.dyscope = dyscope
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        # 可训练参数（TTO 优化目标）
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

        # TTO 相关配置
        self.tto_lr = tto_lr          # 测试时学习率（远小于训练阶段）
        self.tto_steps = tto_steps    # 测试时优化步数（轻量，一般 1-10 步）
        self.tto_loss_type = tto_loss_type  # 优化目标：重建损失/对比损失
        assert tto_loss_type in ['reconstruction', 'contrastive']

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

    def test_time_optimize(self, x, ref_feature=None):
        """
        测试时优化核心方法
        :param x: 测试样本特征 [B, C, H, W]（如 CLIP 提取的特征）
        :param ref_feature: 参考特征（对比损失时需要，如原始高分辨率特征）
        :return: 优化后的上采样特征
        """
        # 1. 仅解冻当前模块的参数（冻结主干模型）
        self.train()  # 切换训练模式（BatchNorm/ Dropout 兼容）
        for param in self.parameters():
            param.requires_grad = True

        # 2. 定义轻量优化器（仅优化 DySample 参数）
        optimizer = optim.Adam(self.parameters(), lr=self.tto_lr)

        # 3. 测试时微调（少量步数）
        for step in range(self.tto_steps):
            optimizer.zero_grad()
            
            # 前向传播
            up_feat = self.forward(x)
            
            # 计算 TTO 损失
            if self.tto_loss_type == 'reconstruction':
                # 重建损失：下采样后与原特征对齐（最通用的轻量目标）
                down_feat = F.adaptive_avg_pool2d(up_feat, (x.shape[2], x.shape[3]))
                loss = F.mse_loss(down_feat, x)
            elif self.tto_loss_type == 'contrastive':
                # 对比损失：与参考特征的余弦相似度最大化（适配 CLIP 等对比学习场景）
                assert ref_feature is not None, "对比损失需要参考特征"
                up_feat_norm = F.normalize(up_feat.flatten(1), dim=1)
                ref_norm = F.normalize(ref_feature.flatten(1), dim=1)
                loss = -torch.mean(torch.sum(up_feat_norm * ref_norm, dim=1))
            
            # 反向传播 + 梯度更新（仅更新 DySample 参数）
            loss.backward()
            optimizer.step()

        # 4. 切换回评估模式，输出优化后的结果
        self.eval()
        with torch.no_grad():
            final_up_feat = self.forward(x)
        return final_up_feat
