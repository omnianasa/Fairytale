import os
import torch.optim as optim
from ImageUtils import ImageUtils
from model import StyleTransferModel


class StyleTransferApp:
    """
    The class is running the full pipeline code
    """

    def __init__(self, style_img_path, imsize=512, content_layers=None, style_layers=None):
        self.imsize = imsize
        self.image_utils = ImageUtils(imsize)
        self.style_img = self.image_utils.load_image(style_img_path)
        self.content_layers = content_layers or ['conv_4']
        self.style_layers = style_layers or ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.model_builder = StyleTransferModel(self.style_img, self.content_layers, self.style_layers)

    def apply_style(self, content_img, num_steps=300, style_weight=1e6, content_weight=1):
        """
        Apply style transfer to a single image (The results could be found in sampleForTest folder)
        """
        model, style_losses, content_losses = self.model_builder.build(content_img)
        input_img = content_img.clone().requires_grad_(True)
        optimizer = optim.LBFGS([input_img])
        run = [0]

        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1) #clamp values between 0:1 to stay imageLike
                optimizer.zero_grad()
                model(input_img)

                style_score = sum(sl.loss for sl in style_losses)
                content_score = sum(cl.loss for cl in content_losses)

                #weighting factors and loss calculations
                style_score *= style_weight
                content_score *= content_weight
                loss = style_score + content_score
                loss.backward()

                #logging
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Step {run[0]}: Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")
                return loss

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)
        return input_img

    def process_batch(self, input_dir, output_dir, frame_prefix="frame_", ext=".jpg", num_frames=26,
                      num_steps=300, style_weight=1e6, content_weight=1):
        """
        Apply style transfer to a batch of images (frames) 
        """
        os.makedirs(output_dir, exist_ok=True)

        for i in range(num_frames):
            frame_name = f"{frame_prefix}{i:04d}{ext}" #frame_0000.jpg
            input_path = os.path.join(input_dir, frame_name)
            output_path = os.path.join(output_dir, f"stylized_{frame_name}")

            print(f"Processing {input_path}...")
            content_img = self.image_utils.load_image(input_path)
            output_img = self.apply_style(content_img, num_steps=num_steps,
                                          style_weight=style_weight, content_weight=content_weight)
            self.image_utils.save_image(output_img, output_path)
            print(f"Saved to {output_path}")


def main():

    # apply style transfer to all frames we got from the video
    app = StyleTransferApp(
        style_img_path="samplesForTest/MemPersistence.jpg",
        imsize=512)

    app.process_batch(
        input_dir="frames",
        output_dir="stylized_frames",
        num_frames=26,
        num_steps=300)


if __name__ == "__main__":
    main()
