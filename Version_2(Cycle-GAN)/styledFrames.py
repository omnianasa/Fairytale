import os
import torch
from torchvision import transforms
from PIL import Image

from components.generator import Generator

class CycleGANVideoStyler:
    def __init__(self, generator_class, model_path, device=None):

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        #generator
        self.G = Generator().to(self.device)
        self.G.load_state_dict(torch.load(model_path, map_location=self.device))
        self.G.eval()

        #transform: normalize to [-1,1] but keep original resolution
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def style_frame(self, img):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            styled_tensor = self.G(img_tensor)
        styled_tensor = (styled_tensor.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
        return transforms.ToPILImage()(styled_tensor)

    def process_frames(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        frame_list = sorted(os.listdir(input_folder))

        for idx, img_name in enumerate(frame_list):
            img_path = os.path.join(input_folder, img_name)
            img = Image.open(img_path).convert("RGB")
            styled_img = self.style_frame(img)
            styled_img.save(os.path.join(output_folder, img_name))
            print(f"[{idx+1}/{len(frame_list)}] Styled: {img_name}")


styler = CycleGANVideoStyler(generator_class=Generator, model_path="G_epoch50.pth")

styler.process_frames("data/data/A", "frames_styled")


