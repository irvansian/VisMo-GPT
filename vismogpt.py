# coding: utf-8

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shutil

import gradio as gr
import random
import torch
import cv2
import re
import uuid
from PIL import Image, ImageDraw, ImageOps, ImageFont
import math
import numpy as np
import argparse
import inspect
import tempfile

from moviepy.config import get_setting
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI, OpenAIChat

from moviepy.editor import *

from ImageTools.imgutils import prompts, get_new_image_name, seed_everything
from ImageTools.instruct_pix2pix import InstructPix2Pix
from ImageTools.text2image import Text2Image
from ImageTools.image_captioning import ImageCaptioning
from ImageTools.image2canny import Image2Canny, CannyText2Image
from ImageTools.image2line import Image2Line, LineText2Image
from ImageTools.image2hed import Image2Hed, HedText2Image
from ImageTools.image2scribble import Image2Scribble, ScribbleText2Image
from ImageTools.image2pose import Image2Pose, PoseText2Image
# from ImageTools.image_segmentation import SegText2Image, Segmenting, ObjectSegmenting
from ImageTools.image2depth import Image2Depth, DepthText2Image
from ImageTools.visual_question_answering import VisualQuestionAnswering
from ImageTools.image2normal import Image2Normal, NormalText2Image
# from ImageTools.image_boxing import Text2Box
from ImageTools.image_inpainting import Inpainting, InfinityOutPainting
# from ImageTools.image_editing import ImageEditing

from VideoTools.video_clipping import VideoClipping
from VideoTools.text2vid import Text2Video
from VideoTools.vid2vid import Video2Video
from VideoTools.vid2frames import Video2Frames
from VideoTools.video_question_answering import VideoDescriptor
from VideoTools.image2video import Image2Video
from VideoTools.video_localizer import SimpleVideoLocalizer
from VideoTools.video_utils import VideoDownload, VideoMetaData

# Grounding DINO
# import groundingdino.datasets.transforms as T
# from groundingdino.models import build_model
# from groundingdino.util import box_ops
# from groundingdino.util.slconfig import SLConfig
# from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
# from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import wget

VISUAL_CHATGPT_PREFIX = """VisMo-GPT is designed to be able to assist with a wide range of text, image and videos related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. VisMo-GPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

VisMo-GPT is able to process and understand large amounts of text, images and videos. As a language model, VisMo-GPT can not directly read images and videos, but it has a list of tools to finish different visual tasks. Each image and video will have a file name formed as "image/xxx.png" or "video/xxx.mp4, and VisMo-GPT can invoke different tools to indirectly understand pictures and videos. When talking about images and videos, VisMo-GPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image and video files, VisMo-GPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image and video. VisMo-GPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image and video content and image and video file name. It will remember to provide the file name from the last tool observation, if a new image or video is generated.

Human may provide new figures to VisMo-GPT with a description. The description helps VisMo-GPT to understand this image or video, but VisMo-GPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, VisMo-GPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

VisMo-GPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since VisMo-GPT is a text language model, VisMo-GPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for VisMo-GPT, VisMo-GPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

VISUAL_CHATGPT_PREFIX_CN = """VisMo-GPT 旨在能够协助完成范围广泛的文本和视觉相关任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。 VisMo-GPT 能够根据收到的输入生成类似人类的文本，使其能够进行听起来自然的对话，并提供连贯且与手头主题相关的响应。

VisMo-GPT 能够处理和理解大量文本和图像。作为一种语言模型，VisMo-GPT 不能直接读取图像，但它有一系列工具来完成不同的视觉任务。每张图片都会有一个文件名，格式为“image/xxx.png”，VisMo-GPT可以调用不同的工具来间接理解图片。在谈论图片时，VisMo-GPT 对文件名的要求非常严格，绝不会伪造不存在的文件。在使用工具生成新的图像文件时，VisMo-GPT也知道图像可能与用户需求不一样，会使用其他视觉问答工具或描述工具来观察真实图像。 VisMo-GPT 能够按顺序使用工具，并且忠于工具观察输出，而不是伪造图像内容和图像文件名。如果生成新图像，它将记得提供上次工具观察的文件名。

Human 可能会向 VisMo-GPT 提供带有描述的新图形。描述帮助 VisMo-GPT 理解这个图像，但 VisMo-GPT 应该使用工具来完成以下任务，而不是直接从描述中想象。有些工具将会返回英文描述，但你对用户的聊天应当采用中文。

总的来说，VisMo-GPT 是一个强大的可视化对话辅助工具，可以帮助处理范围广泛的任务，并提供关于范围广泛的主题的有价值的见解和信息。

工具列表:
------

VisMo-GPT 可以使用这些工具:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN = """用户使用中文和你进行聊天，但是工具的参数应当使用英文。如果要调用工具，你必须遵循如下格式:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

当你不再需要继续调用工具，而是对观察结果进行总结回复时，你必须使用如下格式：


```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX_CN = """你对文件名的正确性非常严格，而且永远不会伪造不存在的文件。

开始!

因为VisMo-GPT是一个文本语言模型，必须使用工具去观察图片而不是依靠想象。
推理想法和观察结果只对VisMo-GPT可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

聊天历史:
{chat_history}

新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""

os.makedirs('image', exist_ok=True)

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)

class ConversationBot:
    def __init__(self, load_dict):
        print(f"Initializing VisMo-GPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})
        
        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        # self.llm = OpenAI(temperature=0)
        self.llm = OpenAIChat(model_name='gpt-4-turbo', temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def init_agent(self, lang):
        self.memory.clear() #clear previous history
        if lang=='English':
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = VISUAL_CHATGPT_PREFIX, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS, VISUAL_CHATGPT_SUFFIX
            place = "Enter text and press enter, or upload an image"
            label_clear = "Clear"
        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = VISUAL_CHATGPT_PREFIX_CN, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN, VISUAL_CHATGPT_SUFFIX_CN
            place = "输入文字并回车，或者上传图片"
            label_clear = "清除"

        # Agent executor
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )
        return gr.update(visible = True), gr.update(visible = False), gr.update(placeholder=place), gr.update(value=label_clear)

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        print(self.agent.agent.llm_chain.prompt)
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        # print(response + "\n")
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state

    def run_image(self, image, state, txt, lang):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        print(image.name)
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models['ImageCaptioning'].inference(image_filename)
        if lang == 'Chinese':
            Human_prompt = f'\nHuman: 提供一张名为 {image_filename}的图片。它的描述是: {description}。 这些信息帮助你理解这个图像，但是你应该使用工具来完成下面的任务，而不是直接从我的描述中想象。 如果你明白了, 说 \"收到\". \n'
            AI_prompt = "收到。  "
        else:
            Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
            AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        # state = state + [(f'<img src="{image_filename}" alt="Image"/>', AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, f'{txt} {image_filename} '

    def run_video(self, video_file, state, txt, lang):
        uid = uuid.uuid4().hex[:8]

        if not os.path.exists('video'):
            os.makedirs('video')

        dest_path = os.path.join('video', f'{uid}.mp4')
        cap = cv2.VideoCapture(video_file.name)

        # Check if the video is horizontal and set new dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width > height:
            target_width = 672
            target_height = 384
        else:
            target_width = width
            target_height = height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dest_path, fourcc, 30.0, (target_width, target_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if width > height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            out.write(frame)

        cap.release()
        out.release()

        self.models['Video2Frames'].inference(dest_path)
        thumbnail = os.path.join('video', 'frames', uid, 'frame_0000.jpg')
        # print("Thumbnail : " + thumbnail)

        Human_prompt = f'\nHuman: provided a video named {dest_path}. If you understand, say \"Received\". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](file={thumbnail})*{dest_path}*", AI_prompt)]
        print(f"\nProcessed run_video, Input video: {dest_path}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")

        return state, state, f'{txt} {dest_path} '

    def run_media(self, file, state, txt, lang):
        filename = file.name
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            return bot.run_image(file, state, txt, lang)
        elif filename.endswith(('.mp4', '.avi', '.mov')):
            return bot.run_video(file, state, txt, lang)
        else:
            raise ValueError("Unsupported file type")


if __name__ == '__main__':
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    print(os.environ.get('OPENAI_API_KEY', 'Environment variable not set'))
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load', type=str, default="ImageCaptioning_cpu, Video2Frames_cpu, VideoClipping_cpu, VisualQuestionAnswering_cpu, VideoDescriptor_cpu, VideoDownload_cpu")
    parser.add_argument('--load', type=str, default="ImageCaptioning_cpu, Video2Frames_cpu, VideoClipping_cpu, VideoDescriptor_cpu, VideoDownload_cpu")
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = ConversationBot(load_dict=load_dict)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        lang = gr.Radio(choices = ['Chinese','English'], value=None, label='Language')
        chatbot = gr.Chatbot(elem_id="chatbot", label="VisMo-GPT")
        state = gr.State([])
        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton(label="🖼️",file_types=["image", "video"])

        lang.change(bot.init_agent, [lang], [input_raws, lang, txt, clear])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_media, [btn, state, txt, lang], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch(share=True, server_name="127.0.0.1", server_port=7861)
