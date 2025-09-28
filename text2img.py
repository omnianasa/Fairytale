import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

class Text2Img:
    def __init__(self, seed=12345, device="cuda", width=512, height=512, inference_num=50, guidance_scale=7.5):
        self.seed = seed
        torch.manual_seed(self.seed)
        self.device = device
        self.width = width
        self.height = height
        self.inference_num = inference_num
        self.guidance_scale = guidance_scale

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        self.pipe.to(self.device)
        # self.pipe.enable_attention_slicing()
        # self.pipe.enable_xformers_memory_efficient_attention()

    def generate(self, prompts_dict, output_dir="./outputs"):
        os.makedirs(output_dir, exist_ok=True)

        for view_name, prompt_text in prompts_dict.items():
            image = self.pipe(
                prompt=prompt_text,
                width=self.width,
                height=self.height,
                num_inference_steps=self.inference_num,
                guidance_scale=self.guidance_scale
            ).images[0]

            save_path = os.path.join(output_dir, f"{view_name}.png")
            image.save(save_path)
            print(f"saved {view_name} view to {save_path}")
