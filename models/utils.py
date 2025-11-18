import random
import numpy as np
import torch
import math
from transformers.generation.configuration_utils import GenerationConfig
from transformers.configuration_utils import PretrainedConfig
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import os

from PIL import Image, ImageDraw, ImageFont




def retrieve_raw_timesteps(
    num_inference_steps: int,
    total_substeps: int,
    final_step_size_scale: float
):
    r"""
    Retrieve the raw times and the number of substeps for each inference step.

    Args:
        num_inference_steps (`int`):
            Number of inference steps.
        total_substeps (`int`):
            Total number of substeps (e.g., 128).
        final_step_size_scale (`float`):
            Scale for the final step size (e.g., 0.5).

    Returns:
        `Tuple[List[float], List[int], int]`: A tuple where the first element is the raw timestep schedule, the second
        element is the number of substeps for each inference step, and the third element is the rounded total number of
        substeps.
    """
    base_segment_size = 1 / (num_inference_steps - 1 + final_step_size_scale)
    raw_timesteps = []
    num_inference_substeps = []
    _raw_t = 1.0
    for i in range(num_inference_steps):
        if i < num_inference_steps - 1:
            segment_size = base_segment_size
        else:
            segment_size = base_segment_size * final_step_size_scale
        _num_inference_substeps = max(round(segment_size * total_substeps), 1)
        num_inference_substeps.append(_num_inference_substeps)
        raw_timesteps.extend(np.linspace(
            _raw_t, _raw_t - segment_size, _num_inference_substeps, endpoint=False).clip(min=0.0).tolist())
        _raw_t = _raw_t - segment_size
    total_substeps = sum(num_inference_substeps)
    return raw_timesteps, num_inference_substeps, total_substeps




def concatenate_images_with_sref(image_lists, style_numbers, label_width_ratio=0.85, font_size=150):
    first_image = image_lists[0][0]
    img_width, img_height = first_image.size
    
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except OSError:
        try:
            font = ImageFont.truetype(f"files/Times_New_Roman_-_Bold.ttf", size=font_size)
        except OSError:
            font = ImageFont.load_default()
    print(font)
    temp_img = Image.new('RGB', (1, 1))
    draw_temp = ImageDraw.Draw(temp_img)


    text_line1 = "--style_code"
    text_line2 = str(style_numbers[0])  
    bbox1 = draw_temp.textbbox((0, 0), text_line1, font=font)
    bbox2 = draw_temp.textbbox((0, 0), text_line2, font=font)
    line_height1 = bbox1[3] - bbox1[1]
    line_height2 = bbox2[3] - bbox2[1]
    text_width = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])

    label_width = max(text_width + 30, int(img_width * label_width_ratio))  

    num_rows = len(image_lists)
    num_cols = len(image_lists[0])
    total_width = label_width + img_width * num_cols
    total_height = img_height * num_rows

    new_image = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(new_image)

    for row_idx, (image_row, number) in enumerate(zip(image_lists, style_numbers)):
        y_offset = row_idx * img_height
        

        total_text_height = line_height1 + line_height2 + 160 
        y_text_start = y_offset + (img_height - total_text_height) // 2
        

        text1_bbox = draw.textbbox((0, 0), text_line1, font=font)
        x_text1 = (label_width - (text1_bbox[2] - text1_bbox[0])) // 2
        y_text1 = y_text_start
        draw.text((x_text1, y_text1), text_line1, fill='black', font=font)
        
        text2 = str(number)
        text2_bbox = draw.textbbox((0, 0), text2, font=font)
        x_text2 = (label_width - (text2_bbox[2] - text2_bbox[0])) // 2
        y_text2 = y_text_start + line_height1 + 80  
        draw.text((x_text2, y_text2), text2, fill='black', font=font)
        

        for col_idx, img in enumerate(image_row):
            top_left_x = label_width + col_idx * img_width
            top_left_y = y_offset
            new_image.paste(img, (top_left_x, top_left_y))

    return new_image
def set_seed(seed: int = 42):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_suppression_coefficient(f, tau, k):
    s = [0] * 1024
    for i in range(1024):
        # breakpoint()
        if f.get(str(i), 0) < tau: 
            s[i] = 1
        else:
            s[i] = math.exp(-k * (f[str(i)] - tau))
    
    return torch.tensor(s)


def load_and_process_config(model_name):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    for attr in ["text_config", "vision_config"]:
        if isinstance(getattr(config, attr, None), dict):
            setattr(config, attr, PretrainedConfig.from_dict(getattr(config, attr)))
    return config


def patched_from_model_config(cls, model_config):
    if not hasattr(model_config, "decoder") or model_config.decoder is None:
        return cls()
    decoder_config = model_config.decoder
    decoder_config_dict = (
        decoder_config.to_dict() if isinstance(decoder_config, PretrainedConfig) else decoder_config
    )
    return cls(**decoder_config_dict)