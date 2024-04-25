import uuid
import replicate
import os
import logging
from urllib.request import urlopen
from ImageTools.imgutils import prompts
class Image2Video:
    def __init__(self, device):
        self.model = "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"
        self.video_length = "25_frames_with_svd_xt"
        self.frames_per_second = 6
        self.sizing_strategy = "maintain_aspect_ratio"
        self.motion_bucket_id = 255
        self.cond_aug = 0.02
        self.decoding_t = 7
        self.seed = 0

    @prompts(name="Generate Video from Image", description= "useful when you want to generate a video from an image and save it to a file. "
                "like: generate a video from 'image_path'."
                "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        try:
            video_filename = os.path.join('video', f"{str(uuid.uuid4())[:8]}.mp4")

            if not os.path.exists('video'):
                os.makedirs('video')

            with open(image_path, "rb") as input_image:
                logging.info("Running model inference with replicate.run")
                video_url = replicate.run(
                    self.model,
                    input={
                        "cond_aug": self.cond_aug,
                        "decoding_t": self.decoding_t,
                        "input_image": input_image,  # Directly pass the file object
                        "video_length": self.video_length,
                        "sizing_strategy": self.sizing_strategy,
                        "motion_bucket_id": self.motion_bucket_id,
                        "frames_per_second": self.frames_per_second,
                    },
                )

                with urlopen(video_url) as response, open(video_filename, "wb") as video_file:
                    video_file.write(response.read())

                return (video_filename, video_filename)
        except Exception as e:
            return (None, f"Unable to generate video from image: {str(e)}")
