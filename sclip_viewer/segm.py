import typing as ty
import gc

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from sclip_viewer.clip_for_segm.pamr import PAMR
from sclip_viewer import clip_for_segm
from sclip_viewer.clip_for_segm.imagenet_template import openai_imagenet_template


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
        self.size = size  # (W, H)

        self.to_tensor_func = transforms.ToTensor()

    def aspect_ratio_preserving_resize(self, img: Image.Image):
        """Resize image to fit in a box (self.size) while preserving aspect ratio."""
        target_w, target_h = self.size
        orig_w, orig_h = img.size

        print(f"{target_w}, {target_h}")
        print(f"{orig_w}, {orig_h}")

        # Calculate scale factors
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        scale = min(scale_w, scale_h)

        # Compute new size
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized_img = img.resize((new_w, new_h), Image.BICUBIC)
        return resized_img

    def __call__(self, image: Image.Image) -> torch.Tensor:

        image_resized = self.aspect_ratio_preserving_resize(image)
        #filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #image_resized.save(f'{filename}.png')
        
        img_tensor = self.to_tensor_func(image_resized)

        if self.rgb_to_bgr:
            img_tensor = img_tensor[[2, 1, 0], :, :]  # RGB -> BGR

        img_tensor = (img_tensor - self.mean) / self.std
        return image_resized, img_tensor
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_for_segm_model, _ = clip_for_segm.load(
    name='ViT-B/16', 
    device=device, 
    jit=False
)
clip_for_segm_model.eval()


class CLIPForSegmentation:
    # def __init__(self, clip_path, name_path, device=torch.device('cuda'),
    #                 pamr_steps=0, pamr_stride=(8, 16), prob_thd=0.0, logit_scale=40, 
    #                 slide_stride=112, slide_crop=224, area_thd=None):
    def __init__(
            self,
            class_names: ty.List[str],
            size: tuple[int, int],
            pamr_steps=4, #4, #0
            pamr_stride=(8, 16), #(4, 8) #(8, 16)
            prob_thd=0.55, #0.65, #0.55, #0.0, 
            logit_scale=90, #95, #90, #85, #65, #40,
            slide_stride=28, #28, #56, #112, 
            slide_crop=224,#224, 
            area_thd=None, #0.05, #0.1, #0.15 #None
            use_template=False
        ):
        
        self.data_preprocessor = CustomSegmDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True,
            size=size
        )
        
        query_words, self.query_idx = get_cls_idx(class_names)
        print(f"Query words: {query_words}")
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                if use_template:
                    query = clip_for_segm.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
                else:
                    query = clip_for_segm.tokenize([qw]).to(device)

                feature = clip_for_segm_model.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)
        print(f"Query features shape: {self.query_features.shape}")
        
        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False

        if pamr_steps > 0:
            self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device)
        else:
            self.pamr = None

    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]

        image_features = clip_for_segm_model.encode_image(img, return_all=True, csa=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features[:, 1:]
        logits = image_features @ self.query_features.T

        patch_size = clip_for_segm_model.visual.patch_size
        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        
        del image_features
        gc.collect()
        torch.cuda.empty_cache()

        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_feature(crop_img)
                preds += nn.functional.pad(
                    crop_seg_logit,
                    (
                        int(x1), int(preds.shape[3] - x2), 
                        int(y1), int(preds.shape[2] - y2))
                    )
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        if self.pamr:
            img = nn.functional.interpolate(img, size=img_size, mode='bilinear')
            logits = self.pamr(img, logits.to(img.dtype)).to(self.dtype)

        del preds, count_mat, crop_seg_logit, crop_img
        gc.collect()
        torch.cuda.empty_cache()

        return logits

    def predict(self, inputs):
        """
        inputs - torch tensor of shape (batch_size, 3, H, W)
        """
        
        with torch.no_grad():
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
            
            if self.slide_crop > 0:
                seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

            seg_preds = self.postprocess_result(seg_logits)

        del seg_logits
        gc.collect()
        torch.cuda.empty_cache()

        return seg_preds
    
    def postprocess_result(self, seg_logits):
        print(f"Seg logits shape: {seg_logits.shape}")
        batch_size = seg_logits.shape[0]
        seg_preds = []
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0) # n_queries * w * h
            
            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            if self.area_thd is not None:
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(seg_logits.dtype)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)  # prone background
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype)          
                seg_logits[1:] *= area_pred.transpose(0, -1)
            
            # flat = seg_logits.reshape(num_cls, -1)      # (num_cls, H*W)
            # class_idx = flat.sum(dim=1).argmax().item()
            # class_inds.append(class_idx)

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0
            
            seg_preds.append(seg_pred)

        return seg_preds
    
    def infer_image(self, image: Image.Image) -> tuple[list[torch.Tensor], Image.Image]:
        image_resized, prep_image = self.data_preprocessor(image)
        tensor = torch.unsqueeze(prep_image, dim=0).to(device)
        print(f"Input tensor shape: {tensor.shape}")
        seg_preds = self.predict(inputs=tensor)
        print(f"Output tensor shape: {len(seg_preds)}")
        print(type(seg_preds[0]))
        return seg_preds, image_resized