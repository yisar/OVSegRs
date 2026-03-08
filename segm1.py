import typing as ty
import gc
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sclip_viewer.clip_for_segm.pamr import PAMR
from sclip_viewer import clip_for_segm
from sclip_viewer.clip_for_segm.imagenet_template import openai_imagenet_template

# --- 辅助函数 ---
def get_cls_idx(name_sets: ty.List[str]) -> tuple[list[str], list[int]]:
    num_cls = len(name_sets)
    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices

class CustomSegmDataPreProcessor:
    def __init__(self, mean, std, rgb_to_bgr=False, size=(2048, 448)):
        self.mean = torch.tensor(mean).view(3, 1, 1) / 255.0
        self.std = torch.tensor(std).view(3, 1, 1) / 255.0
        self.rgb_to_bgr = rgb_to_bgr
        self.size = size 
        self.to_tensor_func = transforms.ToTensor()

    def aspect_ratio_preserving_resize(self, img: Image.Image):
        target_w, target_h = self.size
        orig_w, orig_h = img.size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        return img.resize((new_w, new_h), Image.BICUBIC)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image_resized = self.aspect_ratio_preserving_resize(image)
        img_tensor = self.to_tensor_func(image_resized)
        if self.rgb_to_bgr:
            img_tensor = img_tensor[[2, 1, 0], :, :]
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
            pamr_steps=2,
            pamr_stride=(8, 16),
            prob_thd=0.55,
            logit_scale=90,
            slide_stride=28,
            slide_crop=0,
            area_thd=None,
            use_template=False
        ):
        
        self.data_preprocessor = CustomSegmDataPreProcessor(
            mean=[122.771, 116.746, 104.094], std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True, size=size
        )

        # 1. 加载 AnyUp 模型 (默认关闭 natten 以提高兼容性)
        self.anyup = torch.hub.load('wimmerth/anyup', 'anyup_multi_backbone', use_natten=False).to(device)
        self.anyup.eval()
        
        query_words, self.query_idx = get_cls_idx(class_names)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = clip_for_segm.tokenize([temp(qw) for temp in openai_imagenet_template] if use_template else [qw]).to(device)
                feature = clip_for_segm_model.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        
        self.query_features = torch.cat(query_features, dim=0)
        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device) if pamr_steps > 0 else None

    def forward_feature(self, img, logit_size=None):
        if type(img) == list: img = img[0]

        # 提取 CLIP 特征
        image_features = clip_for_segm_model.encode_image(img, return_all=True, csa=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        patch_features = image_features[:, 1:] 
        patch_size = clip_for_segm_model.visual.patch_size
        w_lr, h_lr = img.shape[-2] // patch_size, img.shape[-1] // patch_size
        lr_feat = patch_features.permute(0, 2, 1).reshape(-1, patch_features.shape[-1], w_lr, h_lr)

        # --- 修复精度冲突：将输入转换为 AnyUp 模型权重相同的类型 (Float) ---
        anyup_dtype = next(self.anyup.parameters()).dtype
        img_input = img.to(anyup_dtype)
        lr_feat = lr_feat.to(anyup_dtype)

        target_size = logit_size if logit_size is not None else img.shape[-2:]
        
        # 2. 调用 AnyUp 进行上采样 (对超大图，增加 q_chunk_size 防止显存溢出)
        hr_feat = self.anyup(img_input, lr_feat, output_size=target_size, q_chunk_size=128)
        
        # 转回原始精度计算 logits
        hr_feat = hr_feat.to(self.dtype)
        logits = torch.einsum('bchw,qc->bqhw', hr_feat, self.query_features)

        del image_features, lr_feat, hr_feat
        gc.collect()
        torch.cuda.empty_cache()
        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        if type(img) == list: img = img[0].unsqueeze(0)
        stride = (stride, stride) if type(stride) == int else stride
        crop_size = (crop_size, crop_size) if type(crop_size) == int else crop_size

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        preds = img.new_zeros((batch_size, self.num_queries, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        
        for h_idx in range(max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
            for w_idx in range(max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
                y2, x2 = min(h_idx * h_stride + h_crop, h_img), min(w_idx * w_stride + w_crop, w_img)
                y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                crop_seg_logit = self.forward_feature(img[:, :, y1:y2, x1:x2])
                preds += nn.functional.pad(crop_seg_logit, (int(x1), int(w_img - x2), int(y1), int(h_img - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        
        preds /= count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        if self.pamr:
            img_hr = nn.functional.interpolate(img, size=img_size, mode='bilinear')
            logits = self.pamr(img_hr, logits.to(img.dtype)).to(self.dtype)
        return logits

    def predict(self, inputs):
        with torch.no_grad():
            batch_img_metas = [dict(ori_shape=inputs.shape[2:])] * inputs.shape[0]
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop) if self.slide_crop > 0 else self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])
            return self.postprocess_result(seg_logits)

    def postprocess_result(self, seg_logits):
        batch_size = seg_logits.shape[0]
        seg_preds = []
        for i in range(batch_size):
            cur_logits = (seg_logits[i] * self.logit_scale).softmax(0)
            if self.num_classes != self.num_queries:
                cls_index = nn.functional.one_hot(self.query_idx, self.num_classes).T.view(self.num_classes, self.num_queries, 1, 1)
                cur_logits = (cur_logits.unsqueeze(0) * cls_index).max(1)[0]
            if self.area_thd is not None:
                predictions = nn.functional.one_hot(cur_logits.argmax(0), self.num_classes).to(cur_logits.dtype)
                area_pred = (predictions[:, :, 1:].sum((0, 1), keepdim=True) > self.area_thd * (h*w)).to(cur_logits.dtype)
                cur_logits[1:] *= area_pred.transpose(0, -1)
            seg_pred = cur_logits.argmax(0, keepdim=True)
            seg_pred[cur_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0
            seg_preds.append(seg_pred)
        return seg_preds

    def infer_image(self, image: Image.Image) -> tuple[list[torch.Tensor], Image.Image]:
        image_resized, prep_image = self.data_preprocessor(image)
        tensor = prep_image.unsqueeze(0).to(device)
        return self.predict(inputs=tensor), image_resized