import os
import uuid

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
from langchain.llms.openai import OpenAIChat
from ImageTools.imgutils import prompts

import openai

class VideoDescriptor:
    def __init__(self, device):
        print("Initializing Video QA")
        self.client = openai.ChatCompletion()

    @prompts(name="Video Question Answering",
             description="useful when you want to ask a question about the video content. "
                         "receives comma separated string of 2,"
                         "represents the video_path and the question to ask about video. "
                         "The output is the answer for the given question.")
    def inference(self, inputs):
        video_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # Calculate the number of frames to skip so that exactly 10 frames are sampled
        if total_frames > 10:
            skip_frames = (total_frames - 1) // 9  # -1 to ensure the last frame is included
        else:
            skip_frames = 1  # In case the video has fewer than 10 frames

        base64Frames = []
        frame_count = 0
        frame_selected = 0

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Check if the current frame count matches the frame to be sampled
            if frame_count == frame_selected * skip_frames:
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                print("Frame", frame_count, "selected.")
                frame_selected += 1
                if frame_selected == 10:  # Stop if we have selected 10 frames
                    break

            frame_count += 1

        video.release()
        print(len(base64Frames), "frames selected for processing.")

        # PROMPT_MESSAGES = [
        #     {
        #         "role": "user",
        #         "content": [
        #             "These are sorted frames from a video that I want to upload. I want to ask, " + question,
        #             *map(lambda x: {"image": x, "resize": 512}, base64Frames[0::50]),
        #         ],
        #     },
        # ]

        params = {
            "model": "gpt-4-vision-preview",
            "messages": self.create_message(image_list=base64Frames, question=question),
            "max_tokens": 800,
        }
        result = self.client.create(**params)
        print("Answer :" + result.choices[0].message.content)
        return result.choices[0].message.content

    def create_message(self, image_list, question):
        message = [
            {"role": "system", "content": "You are video question answering assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "These are sorted frames from a video that I want to upload. I want to ask, " + question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[0]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[1]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[2]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[3]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[4]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[5]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[6]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[7]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[8]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_list[9]}"
                        },
                    },
                ],
            }
        ]
        return message
