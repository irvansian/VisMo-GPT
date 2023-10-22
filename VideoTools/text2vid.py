import os
import shutil
import uuid

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

from ImageTools.imgutils import prompts

class Text2Video:
    def __init__(self, device):
        print(device)
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=self.torch_dtype)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        self.pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
        self.pipe.to(device)

    @prompts(name="Generate Video From Text",
             description="useful when you want to generate a video from a user input text and save it to a file. "
                         "like: generate a video of an object or something, or generate an video that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        video_frames = self.pipe(text, num_inference_steps=40, height=320, width=576, num_frames=36).frames
        temp_video_path = export_to_video(video_frames)
        output_video = os.path.join('video', f"{str(uuid.uuid4())[:8]}.mp4")

        shutil.move(temp_video_path, output_video)
        return output_video
