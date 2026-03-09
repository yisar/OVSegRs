import typing as ty
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sclip_viewer import clip_for_segm
from sclip_viewer.upsample import UPA 

def get_cls_idx(name_sets):
    num_cls = len(name_sets)
    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    return [n.replace('\n', '') for n in class_names], torch.tensor(class_indices)

class CustomSegmDataPreProcessor:
    def __init__(self, mean, std, rgb_to_bgr=False, size=(2048, 448)):
        self.mean = torch.tensor(mean).view(3, 1, 1) / 255.0
        self.std = torch.tensor(std).view(3, 1, 1) / 255.0
        self.rgb_to_bgr = rgb_to_bgr
        self.size = size 

    def __call__(self, image: Image.Image):
        # 保持比例缩放
        tw, th = self.size
        ow, oh = image.size
        scale = min(tw / ow, th / oh)
        image_resized = image.resize((int(ow * scale), int(oh * scale)), Image.BICUBIC)
        
        img_tensor = transforms.ToTensor()(image_resized)
        if self.rgb_to_bgr: img_tensor = img_tensor[[2, 1, 0], :, :]
        img_tensor = (img_tensor - self.mean) / self.std
        return image_resized, img_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_for_segm_model, _ = clip_for_segm.load(name='ViT-B/16', device=device, jit=False)
clip_for_segm_model.eval()

class CLIPForSegmentation:
    def __init__(
            self, 
            class_names: ty.List[str], 
            size: tuple[int, int], 
            prob_thd=0.2, 
            logit_scale=45, 
            slide_stride=112, 
            slide_crop=224, 
            area_thd=None,      # 修正：加上这个参数，解决 TypeError
            use_template=False, # 修正：同步 gradio 中的参数
            cls_token_lambda=-0.3
        ):
        self.data_preprocessor = CustomSegmDataPreProcessor(
            mean=[122.771, 116.746, 104.094], std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True, size=size
        )
        self.current_hr_guide = None 
        self.logit_scale, self.prob_thd = logit_scale, prob_thd
        self.slide_stride, self.slide_crop = slide_stride, slide_crop
        self.cls_token_lambda = cls_token_lambda

        query_words, self.query_idx = get_cls_idx(class_names)
        self.query_idx = self.query_idx.to(device)
        self.num_queries, self.num_classes = len(query_words), self.query_idx.max().item() + 1

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                tokens = clip_for_segm.tokenize([qw]).to(device)
                feat = clip_for_segm_model.encode_text(tokens)
                query_features.append((feat / feat.norm(dim=-1, keepdim=True)).mean(0, keepdim=True))
        self.query_features = torch.cat(query_features, dim=0)

    def forward_feature(self, img_tensor):
        with torch.no_grad():
            image_features = clip_for_segm_model.encode_image(img_tensor, return_all=True, csa=True)
            cls_token, patch_tokens = image_features[:, 0:1, :], image_features[:, 1:, :]
            refined = (patch_tokens + self.cls_token_lambda * cls_token)
            refined /= refined.norm(dim=-1, keepdim=True)
            
            ps = clip_for_segm_model.visual.patch_size
            h_lr, w_lr = img_tensor.shape[-2] // ps, img_tensor.shape[-1] // ps
            lr_feat = refined.permute(0, 2, 1).reshape(1, -1, h_lr, w_lr)

        # 调用 UPA (自带 enable_grad)
        hr_feat = UPA(self.current_hr_guide, lr_feat)
        return torch.einsum('bchw,qc->bqhw', hr_feat.to(self.query_features.dtype), self.query_features)

    def forward_slide(self, img_tensor, img_metas):
        stride, crop = self.slide_stride, self.slide_crop
        full_pil = self.current_hr_guide
        b, _, h_img, w_img = img_tensor.shape
        preds = img_tensor.new_zeros((b, self.num_queries, h_img, w_img))
        count = img_tensor.new_zeros((b, 1, h_img, w_img))
        
        for h_idx in range(max(h_img - crop + stride - 1, 0) // stride + 1):
            for w_idx in range(max(w_img - crop + stride - 1, 0) // stride + 1):
                y2, x2 = min(h_idx * stride + crop, h_img), min(w_idx * stride + crop, w_img)
                y1, x1 = max(y2 - crop, 0), max(x2 - crop, 0)
                
                self.current_hr_guide = full_pil.crop((x1, y1, x2, y2))
                crop_logits = self.forward_feature(img_tensor[:, :, y1:y2, x1:x2])
                preds[:, :, y1:y2, x1:x2] += crop_logits
                count[:, :, y1:y2, x1:x2] += 1
        
        self.current_hr_guide = full_pil
        preds /= count.clamp_min(1.0)
        return F.interpolate(preds, size=img_metas[0]['ori_shape'][:2], mode='bilinear')

    def predict(self, inputs):
        batch_img_metas = [dict(ori_shape=inputs.shape[2:])] * inputs.shape[0]
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas)
        else:
            seg_logits = self.forward_feature(inputs)
            
        with torch.no_grad():
            return self.postprocess_result(seg_logits)

    def postprocess_result(self, seg_logits):
        res = []
        for i in range(seg_logits.shape[0]):
            # 1. 获得概率图 [C, H, W]
            cur = (seg_logits[i] * self.logit_scale).softmax(0)
            
            # 2. 如果类别合并了，处理类别映射
            if self.num_classes != self.num_queries:
                cls_map = F.one_hot(self.query_idx, self.num_classes).T.view(self.num_classes, self.num_queries, 1, 1)
                cur = (cur.unsqueeze(0) * cls_map).max(1)[0]
            
            # 3. 核心修正：确保 p 和 mask 维度完全一致
            p = cur.argmax(0, keepdim=True)  # 得到 [1, H, W]
            
            # 计算最大置信度图 [H, W]
            max_prob, _ = cur.max(0)
            
            # 修正报错：将 p 的第 0 维（通道维）去掉再应用掩码，或者将 mask 扩容
            # 这里我们直接对 p[0] 进行操作，因为它就是 [H, W]
            p[0][max_prob < self.prob_thd] = 0
            
            res.append(p)
        return res

    def infer_image(self, image: Image.Image):
        # 自动旋转修复 (代替 _getexif 判断)
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image).convert("RGB")
            
        image_resized, prep_image = self.data_preprocessor(image)
        self.current_hr_guide = image_resized 
        tensor = prep_image.unsqueeze(0).to(device)
        
        preds = self.predict(inputs=tensor)
        gc.collect(); torch.cuda.empty_cache()
        return preds, image_resized