import os
import uuid

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
# import base64
from langchain.llms.openai import OpenAIChat
from ImageTools.imgutils import prompts

import openai

class VideoDescriptor:
    def __init__(self, device):
        print("Initializing Video QA")
        self.client = openai.ChatCompletion()

    @prompts(name="Video Question Answering",
             description="useful when you want to know what is inside the video. receives video_path as input. "
                         "The input to this tool should be a string, representing the video_path. ")
    def inference(self, video_path):
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        target_fps = 10

        skip_frames = max(int(fps / target_fps), total_frames // (total_frames // 3))

        base64Frames = []
        frame_count = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            if frame_count % skip_frames == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

            frame_count += 1

        video.release()
        print(len(base64Frames), "frames selected for processing.")

        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    "These are frames from a video that I want to upload. It can be single or multiple scene video. Generate a CLIP-like description for every scene.",
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
                ],
            },
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
        }
        result = self.client.create(**params)
        return result.choices[0].message.content
