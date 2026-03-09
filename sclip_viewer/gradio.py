from pprint import pprint
import os, json
from pathlib import Path

import gradio as gr
from PIL import Image

from sclip_viewer.segm import CLIPForSegmentation
from sclip_viewer.visual import *


def segment_image(
        input_image: Image.Image,
        class_names_str: str,
        image_max_width: int, 
        image_max_height: int,
        conf_pixel: float,
        logit_scale: int,
        slide_stride: float,
        slide_crop: float,
        area_thd: float,
        use_template: bool
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
    class_names = class_names_str.split(';')
    class_names = ['background'] + class_names
    if area_thd == 0:
        area_thd = None

    model = CLIPForSegmentation(
        class_names=class_names,
        size=(image_max_width, image_max_height),
        prob_thd=conf_pixel,
        logit_scale=logit_scale,
        slide_stride=slide_stride, 
        slide_crop=slide_crop, 
        area_thd=area_thd,
        use_template=use_template
    )

    map_cls_ind_to_color = get_color_map(class_names)
    pprint(map_cls_ind_to_color)
    classes_legend_image = get_classes_legend_image(class_names, map_cls_ind_to_color)

    input_image = exif_transpose(input_image)
    print(f"Input image shape: {input_image.size}")
    preds_batch, image_resized = model.infer_image(input_image)
    print(f"Resized image shape: {image_resized.size}")
    preds = preds_batch[0]
    print(f"Output tensor shape: {preds.shape}")
    print(f"Unique values of Output tensor: {torch.unique(preds)}")
    
    resulting_mask = get_colored_mask(
        mask_tensor=preds,
        colormap=map_cls_ind_to_color
    )
    print(f"Unique values of resulting mask: {np.unique(resulting_mask)}")

    resulting_image = get_overlay_mask_on_image(
        img=image_resized, 
        mask_tensor=preds,
        colormap=map_cls_ind_to_color,
        alpha=0.5
    )

    return resulting_image, resulting_mask, classes_legend_image


def get_images_paths(folder_path: str) -> list[str]:
    image_extensions = (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")
    return [f"{folder_path}/{file}" for file in os.listdir(folder_path) if file.endswith(image_extensions)]


gradio_examples_path = './sclip_viewer/images'
exmpls_paths = get_images_paths(f"{gradio_examples_path}")
with open(f"{gradio_examples_path}/args_map.json", "r", encoding="utf-8") as f:
        name2args = json.load(f)

examples_values = []
for p in exmpls_paths:
    curr_args = name2args.get(Path(p).name)
    examples_values.append([
        p, 
        curr_args['class_names_str'], 
        curr_args['image_max_width'], 
        curr_args['image_max_height']
    ])


def get_interface():
    with gr.Blocks() as demo:

        gr.Markdown("# Demo: SCLIP")
        with gr.Row():

            with gr.Column():
                
                class_names_str = gr.Textbox(
                    value="car;tree;cropland;grass;house;road;pool",
                    label="Class names (semicolon separated)", 
                    placeholder="cat; purple bird;  ..."
                )
                image_max_width = gr.Number(label="Width (px)", value=512, minimum=448)
                image_max_height = gr.Number(label="Height (px)", value=1024, minimum=448)
                conf_pixel = gr.Slider(0, 1, step=0.01, value=0.5, label="Confidence pixel threshold")
                logit_scale = gr.Slider(1, 200, step=1, value=95, label="Logit scale")
                slide_stride = gr.Slider(28, 280, step=28, value=112, label="Slide stride")
                slide_crop = gr.Slider(112, 672, step=112, value=224, label="Slide crop size")
                area_thd = gr.Slider(0, 0.2, step=0.05, value=0, label="Area threshold")
                use_template = gr.Checkbox(label="Use template")

            input_image = gr.Image(type="pil", label="Input image", width=450, height=650)

            examples = gr.Examples(
                examples=examples_values,
                inputs=[input_image, class_names_str, image_max_width, image_max_height],
                label="Examples",
                examples_per_page=8
            )

        btn = gr.Button("Run")

        with gr.Row():
            out_img = gr.Image(label="Output image")
            out_mask = gr.Image(label="Output mask")
            cls_leg = gr.Image(label="Classes legend")

        btn.click(
            segment_image,
            [
                input_image, class_names_str, 
                image_max_width, image_max_height,
                conf_pixel, logit_scale, 
                slide_stride, slide_crop, area_thd,
                use_template
            ],
            [
                out_img, out_mask, cls_leg
            ]
        )
    
    return demo