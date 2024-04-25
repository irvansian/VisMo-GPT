import os
from ImageTools.imgutils import prompts
from ImageTools.text2image import Text2Image
from VideoTools.image2video import Image2Video


class Text2Video:
    def __init__(self, device):
        self.device = device
        self.text2image = Text2Image(device=device)
        self.image2video = Image2Video(device=device)

    @prompts(name="Generate Video From Text",
             description="Useful when you want to generate a video from a user input text and save it to a file. "
                         "Like: generate a video of an object or something, or generate a video that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image.")
    def inference(self, text):
        # Generate an image from the input text
        image_path = self.text2image.inference(text)

        if not os.path.exists(image_path):
            print("Image generation failed.")
            return None, "Image generation failed."

        video_path, error = self.image2video.inference(image_path)

        if video_path:
            print(f"Generated video saved at: {video_path}")
            return video_path
        else:
            print(error)
            return None, error
