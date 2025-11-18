import os
import torch
from PIL import Image
from models.pipe import CoTylePipeline, PiCoTylePipeline
from io import BytesIO
import requests
from models.vlm_unitok import UniTok
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
import argparse
from models.utils import set_seed, load_and_process_config, patched_from_model_config
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.model import StyleGenerator
import json
from models.model import Qwen2_5_VLForConditionalGeneration_Quant, Qwen2_5_VL_Quant
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from transformers.generation.configuration_utils import GenerationConfig
_original_from_model_config = GenerationConfig.from_model_config
GenerationConfig.from_model_config = classmethod(patched_from_model_config)

def main(args):
    prompt = args.prompt
    output_dir = args.output_path

    unitok_config = {
        'unitok_embed_dim' : 3584,
        'unitok_vocab_width' : 64,
        'unitok_vocab_size' : 1024,
        'unitok_e_temp' : 0.01,
        'unitok_num_codebooks' : 1,
        'unitok_le' : 0.0
    }
    weight_type = torch.bfloat16

    style_generator_path = os.path.join(args.model_path, 'prior')
    config = AutoConfig.from_pretrained(f"{style_generator_path}/config.json")
    style_generator = StyleGenerator._from_config(config)
    state_dict = torch.load(f"{style_generator_path}/prior.pth", map_location='cpu')
    style_generator.load_state_dict(state_dict)
    style_generator.to('cuda', dtype=weight_type)
    
    
    # loading codebook
    unitok = UniTok(unitok_config)
    unitok_state_dict = torch.load(f"{args.model_path}/codebook/model.pth", map_location='cpu')
    unitok.load_state_dict(unitok_state_dict)
    unitok.to('cuda', dtype=weight_type)

    # loading text_encoder
    
    if args.accelerate:
        pipeline = PiCoTylePipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, text_encoder=None,processor=None)
    else:
        pipeline = CoTylePipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, text_encoder=None,processor=None)

    qwen_text_visual_encoder = Qwen2_5_VLForConditionalGeneration_Quant.from_pretrained(
        os.path.join(args.model_path, 'text_encoder'),
    ).to('cuda', dtype=weight_type)

    qwen_text_visual_encoder = Qwen2_5_VL_Quant(unitok, qwen_text_visual_encoder)
    qwen_text_visual_encoder.to('cuda', dtype=weight_type)

    pipeline.text_encoder = qwen_text_visual_encoder
    processor = Qwen2VLProcessor.from_pretrained(os.path.join(args.model_path, 'processor'),
                                                                min_pixels=64 * 28 * 28,
                                                                max_pixels=256 * 28 * 28)
    pipeline.processor = processor
    if args.accelerate:
        adapter_name = pipeline.load_piflow_adapter(  # you may later call `pipe.set_adapters([adapter_name, ...])` to combine other adapters (e.g., style LoRAs)
            'Lakonik/pi-Qwen-Image',
            subfolder='gmqwen_k8_piid_4step',
            target_module_name='transformer')
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(  # use fixed shift=3.2
            pipeline.scheduler.config, shift=3.2, shift_terminal=None, use_dynamic_shifting=False)


    pipeline.to('cuda', dtype=torch.bfloat16)

    pipeline.set_progress_bar_config(disable=None)
    os.makedirs(output_dir, exist_ok=True)
    placeholder_image = Image.new("RGB", (392, 392), (0, 0, 0)) 

    set_seed(args.style_code)
    style_generator_inputs = dict()
    style_generator_inputs['input_ids'] = torch.randint(low=0, high=1024, size=(1, 1)).to('cuda')
    style_generator_inputs['attention_mask'] = torch.ones(style_generator_inputs['input_ids'].shape).to('cuda')
    

    with open(f'{args.model_path}/freq.json', 'r') as f:
        code_freq = json.load(f)
    generated_ids = style_generator.generate(
        **style_generator_inputs,
        max_new_tokens=195,
        temperature=1.0,  # Increased from 0.7
        top_k=200,         # Added top_k sampling
        top_p=0.95,       # Added nucleus sampling
        do_sample=True ,   # Enable sampling
        repetition_penalty=50.0,
        code_freq=code_freq,
        code_freq_threshold=args.freq_threshold,
        k=args.freq_k,
        )

    if args.accelerate:
        sample_steps = 4
    else:
        sample_steps = 40

    set_seed(args.seed)
    inputs = {
        "image": [placeholder_image],
        "prompt": prompt,
        "generator": torch.manual_seed(args.seed),
        "true_cfg_scale": 6.0,
        "negative_prompt": "丑陋，怪物，怪兽，畸形，变异，结构不合理，肢体不合理，人脸扭曲, 肢体错乱,突兀",
        "num_inference_steps": sample_steps,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
        "codebook_id": generated_ids,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)

    for idx, res in enumerate(output.images):
        res.save(f"{output_dir}/{args.style_code}.png")
        print(f'The result is saved to f"{output_dir}/{args.style_code}.png"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--style_code",
        type=int,
        default=1234567,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default='./pretrained_models',
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs",
    )


    parser.add_argument(
        "--prompt",
        type=str,
        default="A lovely crystal snake spirit, slender and nimble, wears an exquisite crystal crown atop its head. Its scales are translucent, shimmering like crystal, its eyes are bright and round, and its expression is lively. Its body coils naturally, its tail gracefully curved, its overall posture harmonious and beautiful.",
    )
    parser.add_argument(
        "--freq_threshold",
        type=int,
        default=90000,
    )
    parser.add_argument(
        "--freq_k",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--accelerate",
        action='store_true'
    )

    args = parser.parse_args()
    main(args)