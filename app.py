print("Starting CoTyle app...")
import os
import subprocess
from pathlib import Path

import sys
import json
import torch
import gradio as gr
import spaces
from PIL import Image
from huggingface_hub import snapshot_download
import gc
import psutil
from functools import partial



REPO_ID = "Kwai-Kolors/CoTyle"
HF_TOKEN = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"
weight_type = torch.bfloat16 if device == "cuda" else torch.float32

SUGGESTED_PROMPTS = [
    "An artist sits outdoors, engrossed in their work, brush in hand, capturing the scene with focused intensity. On the canvas, trees and buildings blend seamlessly with the real-world surroundings. Symbols from different cultures, along with animals, plants, and abstract lines, float around them. As the brush touches the canvas, the paint transforms into points of light that scatter, while sheets of paper and flower petals flutter in the air, creating a sense of movement. The atmosphere is a high-detail fusion of art and reality.",
    "Seagulls soar along the seaside under the setting sun, as a couple in wedding attire holds hands.",
    "A cute, chubby werewolf holds a balloon and candy, looking adorably mischievous. The background features a full moon on a night sky.",
    "A classical beauty, dressed in a dreamy, light pink flowing gown with wide sleeves, adorned with countless tiny wind crystals.",
    "The train sped swiftly across a large bridge.",
    "In front of the door stands an apple tree with two apples glistening with dewdrops. A beautiful little bird with vibrant feathers perches on a branch, displaying intricate textures and clear details.",
]

# 预设模板配置
PRESET_TEMPLATES = [
    {
        "name": "--sref 1234567",
        "image_path": "assets/1234567.jpg",
        "style_code": 1234567,
        "seed": 42,
        "prompts": [
            "An artist sits outdoors, engrossed in their work, brush in hand, capturing the scene with focused intensity. On the canvas, trees and buildings blend seamlessly with the real-world surroundings. Symbols from different cultures, along with animals, plants, and abstract lines, float around them. As the brush touches the canvas, the paint transforms into points of light that scatter, while sheets of paper and flower petals flutter in the air, creating a sense of movement. The atmosphere is a high-detail fusion of art and reality.",
            "Seagulls soar along the seaside under the setting sun, as a couple in wedding attire holds hands.",
            "A cute, chubby werewolf holds a balloon and candy, looking adorably mischievous. The background features a full moon on a night sky.",
            "A classical beauty, dressed in a dreamy, light pink flowing gown with wide sleeves, adorned with countless tiny wind crystals.",
        ]
    },
    {
        "name": "--sref 666666666",
        "image_path": "assets/666666666.jpg",
        "style_code": 666666666,
        "seed": 42,
        "prompts": [
            "A chubby, white, curly-furred baby lamb in anime style, with a pink nose and short mouth, stands on grass looking directly at the camera.",
            "A boy with a backpack stands on a mountain peak, bathed in sunlight, with continuous mountain ranges in the background.",
            "Aerial view: distant wind turbines, mountains, a river, heavy snowfall, and four or five people in orange work uniforms and white safety helmets marching in a line through the snow.",
            "A beautiful Chinese woman in ancient red silk attire rides a white horse, holding a red tassel spear, facing an enemy army of thousands; ethereal clouds swirl around her, and behind her stand countless celestial soldiers clad in white armor; documentary photography style."
        ]
    },
    {
        "name": "--sref 886",
        "image_path": "assets/886.jpg",
        "style_code": 886,
        "seed": 42,
        "prompts": [
            "A lovely crystal snake spirit, slender and nimble, wears an exquisite crystal crown atop its head. Its scales are translucent, shimmering like crystal, its eyes are bright and round, and its expression is lively. Its body coils naturally, its tail gracefully curved, its overall posture harmonious and beautiful.",
            "Seagulls soar along the seaside under the setting sun, as a couple in wedding attire holds hands.",
            "A cute, chubby werewolf holds a balloon and candy, looking adorably mischievous. The background features a full moon on a night sky.",
            "The train sped swiftly across a large bridge."
        ]
    },
    {
        "name": "--sref 10241024",
        "image_path": "assets/10241024.jpg",
        "style_code": 10241024,
        "seed": 42,
        "prompts": [
            "An elegant tabby cat steps gracefully through the doorway, its soft paws landing silently on the floor. Its amber eyes scan the surroundings with keen alertness, taking in every detail of the room.",
            "Mickey Mouse appears in the 1920s gangster world, dressed in a long trench coat and a fedora, holding an old-fashioned revolver. The backdrop is a dimly lit Chicago alleyway, where shadows stretch across the cobblestones and the air is thick with the intrigue of the era.",
            "A motorcycle speeds down the highway, the rider clad in black leather, with a biker girl seated behind him. The setting sun glints off the metallic fuel tank, while the rear wheel kicks up a trail of dust. In the background, the desolate road stretches endlessly towards the horizon, framed by the vast wilderness.",
            "A classical beauty, dressed in a dreamy, light pink flowing gown with wide sleeves, adorned with countless tiny wind crystals."
        ]
    },
    {
        "name": "--sref 4396",
        "image_path": "assets/4396.jpg",
        "style_code": 4396,
        "seed": 42,
        "prompts": [
            "A boy and a girl are walking along the lakeside, surrounded by vibrant flowers, lush grass, and verdant trees.",
            "A hazy full moon hangs high in the night sky, with the bustling streets of an ancient town below, adorned with a variety of lanterns that are vibrant and bright.",
            "A cartoon bear with a wide, round mouth and neatly arranged teeth, illustration, mascot, chubby.",
            "A real-life depiction of a warrior goddess is strikingly beautiful, adorned in metallic armor. She has long legs and sports enormous wings, adding to her majestic presence. A crown sits atop her head, and she wields a weapon, poised in a dynamic battle stance."
        ]
    },
]



def load_models():
    global pipeline, style_generator, unitok, processor, code_freq, local_repo_dir
    if "pipeline" in globals():
        return
    print('='*10, 'before download')
    local_repo_dir = snapshot_download(
        repo_id=REPO_ID,
        token=HF_TOKEN,
        allow_patterns=[
            "prior/**",
            "codebook/**",
            "tokenizer/**",
            "processor/**",
            "text_encoder/**",
            "freq.json",
            "transformer/**",
            "vae/**",
            "*.json",
            "*.pth",
            "*.safetensors",
        ],
        resume_download=True,
    )
    print('='*10, 'after download')
    sys.path.append(".")
    from models.pipe import CoTylePipeline, PiCoTylePipeline
    from models.vlm_unitok import UniTok
    from models.model import StyleGenerator, Qwen2_5_VLForConditionalGeneration_Quant, Qwen2_5_VL_Quant
    from models.utils import patched_from_model_config
    from transformers import Qwen2VLProcessor, AutoConfig
    from transformers.generation.configuration_utils import GenerationConfig
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    GenerationConfig.from_model_config = classmethod(patched_from_model_config)
    unitok_config = {
        "unitok_embed_dim": 3584,
        "unitok_vocab_width": 64,
        "unitok_vocab_size": 1024,
        "unitok_e_temp": 0.01,
        "unitok_num_codebooks": 1,
        "unitok_le": 0.0,
    }
    style_generator_path = os.path.join(local_repo_dir, "prior")
    config = AutoConfig.from_pretrained(style_generator_path)
    style_generator = StyleGenerator._from_config(config)
    state_dict = torch.load(os.path.join(style_generator_path, "prior.pth"), map_location="cpu")
    style_generator.load_state_dict(state_dict)
    style_generator.to(device, dtype=weight_type)
    codebook_path = os.path.join(local_repo_dir, "codebook")
    unitok = UniTok(unitok_config)
    unitok_state_dict = torch.load(os.path.join(codebook_path, "model.pth"), map_location="cpu")
    unitok.load_state_dict(unitok_state_dict)
    unitok.to(device, dtype=weight_type)

    pipeline = PiCoTylePipeline.from_pretrained(
        local_repo_dir,
        torch_dtype=weight_type,
        text_encoder=None,
        processor=None,
        safety_checker=None,
        requires_safety_checker=False,
    )

    qwen_text_visual_encoder = Qwen2_5_VLForConditionalGeneration_Quant.from_pretrained(
        local_repo_dir,
        subfolder="text_encoder",
    ).to(device, dtype=weight_type)
    qwen_text_visual_encoder = Qwen2_5_VL_Quant(unitok, qwen_text_visual_encoder)
    qwen_text_visual_encoder.to(device, dtype=weight_type)
    pipeline.text_encoder = qwen_text_visual_encoder
    processor = Qwen2VLProcessor.from_pretrained(
        local_repo_dir,
        subfolder="processor",
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    pipeline.processor = processor
    adapter_name = pipeline.load_piflow_adapter(  # you may later call `pipe.set_adapters([adapter_name, ...])` to combine other adapters (e.g., style LoRAs)
        'Lakonik/pi-Qwen-Image',
        subfolder='gmqwen_k8_piid_4step',
        target_module_name='transformer')
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(  # use fixed shift=3.2
        pipeline.scheduler.config, shift=3.2, shift_terminal=None, use_dynamic_shifting=False)



    pipeline.to(device, dtype=weight_type)
    pipeline.set_progress_bar_config(disable=True)

    with open(os.path.join(local_repo_dir, "freq.json"), "r") as f:
        code_freq = json.load(f)
    print('='*10, " All models loaded successfully!")

@spaces.GPU
def generate_images(style_code, seed, num_prompts, *args):
    try:
        style_code = int(style_code)
    except Exception:
        style_code = 0
    try:
        seed = int(seed)
    except Exception:
        seed = 42
    try:
        num_prompts = int(num_prompts)
    except Exception:
        num_prompts = 1

    load_models()
    from models.utils import set_seed

    prompts = []
    for i in range(num_prompts):
        if i < len(args):
            prompt_text = (args[i] or "").strip()
            if prompt_text:
                prompts.append(prompt_text)

    if not prompts:
        raise gr.Error("Please enter at least one valid prompt!")

    set_seed(style_code)
    style_generator_inputs = {
        "input_ids": torch.randint(low=0, high=1024, size=(1, 1)).to(device),
        "attention_mask": torch.ones((1, 1)).to(device),
    }
    with torch.no_grad():
        generated_ids = style_generator.generate(
            **style_generator_inputs,
            max_new_tokens=195,
            temperature=1.0,
            top_k=200,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=50.0,
            code_freq=code_freq,
            code_freq_threshold=90000,
            k=0.0001,
        )
    placeholder_image = Image.new("RGB", (392, 392), (0, 0, 0))
    results = []
    for i, prompt in enumerate(prompts):

        set_seed(seed)
        inputs = {
            "image": [placeholder_image],
            "prompt": prompt,
            "generator": torch.Generator(device=device).manual_seed(seed),
            "true_cfg_scale": 6.0,
            "negative_prompt": "ugly, monster, grotesque, deformed, mutated, anatomically incorrect, distorted face, disfigured limbs, unnatural posture, blurry, low quality",
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
            "codebook_id": generated_ids,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"generating {i+1}/{len(prompts)}")
        with torch.inference_mode():
            output = pipeline(**inputs)
        results.append(output.images[0])
        print(f"Successfully generate {i+1}/{len(prompts)} images")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    del output

    print(f"Successfully generate {len(results)} images")
    return results

def load_preset_template(template_idx):
    template = PRESET_TEMPLATES[template_idx]

    outputs = [
        template["style_code"],
        template["seed"],
        4,
    ]

    for i in range(4):
        outputs.append(template["prompts"][i])

    for i in range(2):
        outputs.append(SUGGESTED_PROMPTS[4+i])

    return tuple(outputs)

def create_placeholder_image(text):
    return Image.new('RGB', (300, 200), color=(240, 240, 240))


custom_js = """
function() {
    // 优化 Gradio 的更新性能
    const style = document.createElement('style');
    style.textContent = `
        .gradio-container { transition: none !important; }
        .gr-box { transition: none !important; }
    `;
    document.head.appendChild(style);
}
"""

with gr.Blocks(
    # theme=gr.themes.midnight(),
    theme = 'Taithrah/Minimal',
    js=custom_js,  # 添加自定义 JS 来禁用不必要的动画
    css="""
    .prompt-hint {
        font-size: 0.9em;
        color: #666;
        margin-top: -8px;
        margin-bottom: 12px;
    }
    .preset-container {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        background: white;
        height: 100%;
        display: flex;
        flex-direction: column;
        max-width: 280px;
        margin: 0 auto;
    }
    .preset-container:hover {
        border-color: #2196F3;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);
        transform: translateY(-2px);
    }
    .preset-image-container {
        width: 100%;
        height: 240px;
        overflow: hidden;
        border-radius: 8px;
        margin-bottom: 1px;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .preset-image-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .preset-text {
        text-align: center;
        font-weight: bold;
        font-size: 1.0em;
        color: #333;
        padding: 3px 0;
    }
    .preset-row {
        margin-bottom: 10px;
        justify-content: center;
        gap: 15px;
    }
    .preset-section {
        max-width: 1900px;
        margin: 0 auto;
        padding: 0 20px;
    }

    .gr-box, .gr-form, .gr-input {
        transition: none !important;
    }


    .gradio-gallery {
        width: 100% !important;
        max-width: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .gradio-gallery .grid-container {
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)) !important;
        gap: 12px !important;
    }


    .form-label {
        color: #2d3748 !important;
        font-weight: 600;
    }
"""
) as demo:
    gr.HTML(
        """
    <div align="center" style="font-size: 40px;">
    🎨 CoTyle: Unlocking Code-to-Style Image Generation with Discrete Style Space
    <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin: 15px 0;">
    <a href="https://arxiv.org/abs/2511.10555"><img alt="Build" src="https://img.shields.io/badge/arXiv-Paper-da282a.svg"></a>
    <a href="https://Kwai-Kolors.github.io/CoTyle/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-Homepage-yellow"></a>
    <a href="https://github.com/Kwai-Kolors/CoTyle"><img alt="Build" src="https://img.shields.io/badge/GitHub-Code-f8f0f0.svg"></a>
    <a href="https://huggingface.co/spaces/Kwai-Kolors/CoTyle"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-32CD32"></a>
    <a href="https://huggingface.co/Kwai-Kolors/CoTyle"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-fd8b02"></a>
    </div>
    </div>
        """
    )

    with gr.Row():
        with gr.Column():
            # style_code = gr.Number(label="Style Code", value=1234567, step=1)
            style_code = gr.Slider(
                minimum=1,
                maximum=4294967296,
                value=1234567,
                step=1,
                label="Style Code",
            )


            num_prompts = gr.Slider(
                minimum=1,
                maximum=6,
                value=4,
                step=1,
                label="Number of Prompts (You can choose how many prompt images to generate at once)",
            )


            text_inputs = []
            for i in range(6):
                default_prompt = SUGGESTED_PROMPTS[i] if i < len(SUGGESTED_PROMPTS) else ""
                textbox = gr.Textbox(
                    value=default_prompt,
                    label=f"Prompt {i+1}",
                    lines=3,
                    max_lines=10,
                    placeholder="Enter your prompt here...",
                    visible=(i < 4),
                )
                text_inputs.append(textbox)

            seed = gr.Slider(
                minimum=1,
                maximum=4294967296,
                value=42,
                step=1,
                label="Seed",
            )

            run_btn = gr.Button("✨ Generate All Images", variant="primary", size="lg")


        with gr.Column():
            gallery = gr.Gallery(
                label="Generated Results",
                show_label=True,
                columns=2,
                object_fit="contain",
                height="100%",
            )
            gr.Markdown(
                """
                > ⚠️ <strong>Note</strong>:
                > - The Gradio apps use an accelerated version, which may result in a slight reduction in image generation quality.
                > - This demo is the open-source version, utilizing [Qwen-Image](https://github.com/QwenLM/Qwen-Image) as the pre-trained model, while the more powerful closed-source version employs Kolors as the pre-trained model and will soon be launched on the [KlingAI](https://app.klingai.com/global/?gad_source=1&gad_campaignid=22803840655&gbraid=0AAAAA_AcKMnNNjEHRRI1l9_5z1qK881dO).
                """
                )

            gr.Markdown(
                """
            > ✅ <strong>Tips</strong>:
            > - Adjust the <strong>Number of Prompts</strong> slider to add or remove input rows.
            > - Type your own prompts directly in the text boxes .
            > - You can click any template below to quickly load preset style code and prompts.
            """
            )

    def update_textboxes_visibility(n):
        return [gr.update(visible=(i < n)) for i in range(6)]

    num_prompts.change(
        fn=update_textboxes_visibility,
        inputs=num_prompts,
        outputs=text_inputs,
        queue=False,
    )

    input_components = [style_code, seed, num_prompts] + text_inputs

    run_btn.click(
        fn=generate_images,
        inputs=input_components,
        outputs=gallery,
    )


    output_components = [style_code, seed, num_prompts] + text_inputs
    with gr.Column(elem_classes="preset-section"):
        gr.Markdown("## 🎯 Examples")
        gr.Markdown("Click any example below to quickly load preset style code, seed, and prompts")


        with gr.Row(elem_classes="preset-row"):
            for i in range(5):
                with gr.Column(scale=1, min_width=250):
                    template = PRESET_TEMPLATES[i]
                    with gr.Column(elem_classes="preset-container"):
                        if os.path.exists(template["image_path"]):
                            preset_img = gr.Image(
                                value=template["image_path"],
                                show_label=False,
                                interactive=False,
                                container=False,
                                height=280,
                                elem_classes="preset-image-container"
                            )
                        else:
                            placeholder = create_placeholder_image(template["name"])
                            preset_img = gr.Image(
                                value=placeholder,
                                show_label=False,
                                interactive=False,
                                container=False,
                                height=280,
                                elem_classes="preset-image-container"
                            )

                        preset_btn = gr.Button(
                            value=template["name"],
                            variant="secondary",
                            size="lg"
                        )


                        preset_btn.click(
                            fn=partial(load_preset_template, i),
                            inputs=None,
                            outputs=output_components,
                            queue=False,
                        )
if __name__ == "__main__":
    load_models()
    demo.queue(max_size=1).launch(
        max_threads=1,
        share=True
    )
