import io
import requests
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
from PIL import Image
from torchvision import transforms
import warnings
from sclip_viewer.upsample import UPA 


warnings.filterwarnings('ignore')

# ===================== 1. 基础配置 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = 448  # 高分辨率引导图尺寸
clip_input_size = 224  # CLIP B/16 的标准预处理尺寸

# ===================== 3. 加载模型与图像 =====================
model_name = "ViT-B-16"
pretrained = "laion2b_s34b_b88k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
model.eval()
visual = model.visual

# 加载图像
url = "https://dd-static.jd.com/ddimgp/jfs/t20270412/402218/6/3083/4812860/69ad6837Fb5f43e3d/0936bb8fa04cf2c5.jpg"
try:
    resp = requests.get(url, timeout=5)
    raw_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
except Exception as e:
    print(f"图像加载失败: {e}，使用灰色占位图")
    raw_img = Image.new('RGB', (img_size, img_size), color='gray')

# ===================== 4. 图像预处理 =====================
# 修正：HR 引导图直接用 PIL 图像（调整尺寸即可，UPA 内部会处理张量转换）
hr_guide = raw_img.resize((img_size, img_size))  # 传给 UPA 的是 PIL Image

# CLIP 标准输入
clip_img = preprocess(raw_img).unsqueeze(0).to(device)

# ===================== 5. 提取 CLIP 特征 (低分辨率) =====================
with torch.no_grad():
    # 1. 卷积投影
    x = visual.conv1(clip_img)  # [1, 768, 14, 14]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [1, 196, 768]
    
    # 2. 拼接 Class Token
    cls_token = visual.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, x], dim=1)  # [1, 197, 768]
    
    # 3. 加上位置编码
    pos_emb = visual.positional_embedding.to(x.dtype)
    x = x + pos_emb
    
    # 4. Transformer 传播
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = visual.ln_post(x)
    
    # 5. 取出空间 Tokens 并还原为特征图 [1, 768, 14, 14]
    tokens = x[:, 1:, :]
    B, N, C = tokens.shape
    h = w = int(N ** 0.5)
    lr_feat = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

# ===================== 6. 调用 UPA 采样器 (核心修改) =====================
# 即使 UPA 内部有 enable_grad，通常可视化时我们只需要推断结果
# 如果 UPA 依赖于梯度迭代（如某些优化对齐方法），则不要包裹 no_grad
# 否则建议包裹以节省显存
with torch.set_grad_enabled(False): 
    # 修正：第一个参数传 PIL Image (hr_guide)，而非张量
    hr_feat = UPA(hr_guide, lr_feat) 

# ===================== 7. 可视化逻辑 (PCA) =====================
def get_pca_rgb(feats):
    """将高维特征通过 PCA 降维到 3 维用于 RGB 可视化"""
    # feats shape: [1, C, H, W]
    b, c, h, w = feats.shape
    f = feats[0].permute(1, 2, 0).reshape(-1, c) # [H*W, C]
    f = f - f.mean(0)
    
    # 使用奇异值分解进行 PCA
    U, S, V = torch.pca_lowrank(f, q=3)
    proj = f @ V[:, :3] # 取前三个主成分
    
    # 归一化到 [0, 1]
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    return proj.reshape(h, w, 3).cpu().numpy()

# 计算低分辨率和高分辨率的 PCA 结果
lr_rgb = get_pca_rgb(lr_feat)
hr_rgb = get_pca_rgb(hr_feat)

# ===================== 8. 绘图展示 =====================
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.imshow(lr_rgb)
plt.title(f"Original CLIP Features ({h}x{w})", fontsize=12)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hr_rgb)
plt.title(f"UPA Refined Features ({hr_feat.shape[2]}x{hr_feat.shape[3]})", fontsize=12)
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"处理完成！LR 特征尺寸: {lr_feat.shape}, HR 特征尺寸: {hr_feat.shape}")