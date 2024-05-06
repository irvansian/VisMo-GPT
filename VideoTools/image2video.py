import uuid
import replicate
import os
import logging
from urllib.request import urlopen
from ImageTools.imgutils import prompts
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch
from PIL import Image
import cv2
import numpy as np


class Image2Video:
    def __init__(self, device):
        # self.model = "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"
        # self.video_length = "25_frames_with_svd_xt"
        # self.frames_per_second = 6
        # self.sizing_strategy = "maintain_aspect_ratio"
        # self.motion_bucket_id = 255
        # self.cond_aug = 0.02
        # self.decoding_t = 7
        # self.seed = 0
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        )
        self.pipe.enable_model_cpu_offload()

    @prompts(name="Generate Video from Image",
             description="useful when you want to generate a video from an image and save it to a file. "
                         "like: generate a video from 'image_path'."
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        # try:
        #     video_filename = os.path.join('video', f"{str(uuid.uuid4())[:8]}.mp4")
        #
        #     if not os.path.exists('video'):
        #         os.makedirs('video')
        #
        #     with open(image_path, "rb") as input_image:
        #         print("Running model inference with replicate.run")
        #         video_url = replicate.run(
        #             self.model,
        #             input={
        #                 "cond_aug": self.cond_aug,
        #                 "decoding_t": self.decoding_t,
        #                 "input_image": input_image,
        #                 "video_length": self.video_length,
        #                 "sizing_strategy": self.sizing_strategy,
        #                 "motion_bucket_id": self.motion_bucket_id,
        #                 "frames_per_second": self.frames_per_second,
        #             },
        #         )
        #
        #         with urlopen(video_url) as response, open(video_filename, "wb") as video_file:
        #             video_file.write(response.read())
        #
        #     return (video_filename, video_filename)
        # except Exception as e:
        #     return (None, f"Unable to generate video from image: {str(e)}")
        image = Image.open(image_path)
        image = image.resize((1024, 576))

        generator = torch.manual_seed(42)
        print(torch.cuda.get_device_properties(0).total_memory)
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())
        frames = self.pipe(image, decode_chunk_size=8, generator=generator).frames[0]
        video_filename = os.path.join('video', f"{str(uuid.uuid4())[:8]}.mp4")
        export_to_video(frames, video_filename, fps=7)

        return interpolate_video(video_filename)


def interpolate_frames(frame1, frame2, num_interpolations):
    return [(frame1 * (1 - alpha) + frame2 * alpha).astype(np.uint8)
            for alpha in np.linspace(0, 1, num_interpolations + 2)[1:-1]]


def interpolate_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output = os.path.join('video', f"{str(uuid.uuid4())[:8]}.mp4")
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    prev_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if prev_frame is not None:
                num_interpolations = fps // 7 - 1
                interpolated_frames = interpolate_frames(prev_frame, frame, num_interpolations)
                for interp_frame in interpolated_frames:
                    out.write(interp_frame)

            out.write(frame)
            prev_frame = frame
    finally:
        cap.release()
        out.release()

    return output
