from PIL import Image, ImageDraw, ImageFont, ExifTags
import numpy as np
import torch
from colorsys import hsv_to_rgb


def get_color_map(classes_names: list[str]) -> dict[str, tuple[int, int, int]]:
    n_classes = len(classes_names)
    rng = np.random.RandomState(42)
    map_cls_ind_to_color = {}
    for idx in range(n_classes):
        if idx == 0:
            map_cls_ind_to_color[idx] = (0, 0, 0)
        else:
            h = (0.11 + idx * 0.61803398875) % 1.0  # golden ratio step
            r, g, b = hsv_to_rgb(h, 0.75, 0.92)
            map_cls_ind_to_color[idx] = (int(r * 255), int(g * 255), int(b * 255))
            #map_cls_ind_to_color[idx] = tuple(int(x) for x in rng.randint(0, 256, size=3))
    
    return map_cls_ind_to_color


def get_classes_legend_image(
        classes_names: list[str],
        map_cls_ind_to_color: dict[str, tuple[int, int, int]]
    ) -> Image.Image:
    classes_names = {ind:cls for ind, cls in enumerate(classes_names)}

    rect_w, rect_h = 40, 40
    padding = 10
    font_size = 24

    width = rect_w + 3 * padding + max(len(label) for label in classes_names.values()) * (font_size // 2)
    height = len(classes_names) * (rect_h + padding) + padding

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()
    for i, (cls_idx, label) in enumerate(classes_names.items()):
        y = padding + i * (rect_h + padding)
        draw.rectangle([padding, y, padding + rect_w, y + rect_h], fill=map_cls_ind_to_color[cls_idx])
        draw.text((padding * 2 + rect_w, y + (rect_h - font_size) // 2), label, fill="black", font=font)

    return img


def get_colored_mask(
        mask_tensor: torch.Tensor,
        colormap: dict[int, tuple[int,int,int]] | None = None
    ) -> Image.Image:
    """
    Args:
        mask_tensor: torch.Tensor of shape (1, H, W) or (H, W), with integer labels:
            0 = background, 1..N = classes in class_names.
        class_names: list of class names in order [‘cat’, ‘bird’, …].
        save_path: where to save the PNG (e.g. 'mask_colored.png').
        colormap: optional dict mapping label→RGB; if None, a default is generated.
    """
    if mask_tensor.dim() == 3:
        mask = mask_tensor.squeeze(0)
    else:
        mask = mask_tensor
    mask_np = mask.detach().cpu().numpy()

    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, color in colormap.items():
        color_mask[mask_np == lbl] = color
    
    res = Image.fromarray(color_mask)
    return res


def get_overlay_mask_on_image(
    img: Image.Image,
    mask_tensor: torch.Tensor,
    colormap: dict[int, tuple[int, int, int]],
    alpha: float = 0.5
) -> Image.Image:
    """
    Overlays a semi-transparent color mask on the original image.
    Mask is only applied to non-background pixels (class 0 is background).
    """
    # 1) Squeeze out any batch/channel dims
    if mask_tensor.dim() == 3:
        mask = mask_tensor.squeeze(0)
    else:
        mask = mask_tensor

    # 2) Bring mask to CPU numpy (H_mask x W_mask)
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)

    # 3) Convert PIL image to numpy (H_img x W_img x 3)
    img_np = np.array(img.convert("RGB"), copy=True)
    h_img, w_img = img_np.shape[:2]

    # 4) If mask and image shapes differ, resize mask with nearest neighbor
    if mask_np.shape != (h_img, w_img):
        mask_img = Image.fromarray(mask_np, mode="L")
        mask_img = mask_img.resize((w_img, h_img), resample=Image.NEAREST)
        mask_np = np.array(mask_img)

    # 5) Prepare output and overlay
    out = img_np.copy()
    for lbl, color in colormap.items():
        if lbl == 0:
            continue  # skip background
        mask_area = (mask_np == lbl)
        # Blend: α·color + (1−α)·original
        blended = (
            alpha * np.array(color)[None, None, :]
            + (1 - alpha) * img_np[mask_area]
        ).astype(np.uint8)
        out[mask_area] = blended

    res = Image.fromarray(out)
    return res


def exif_transpose(img: Image.Image) -> Image.Image:
    try:
        exif = img._getexif()
        if exif is None:
            return img
        exif = dict(exif.items())
        orientation = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation = k
                break
        if orientation and orientation in exif:
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"EXIF correction failed: {e}")
    return img