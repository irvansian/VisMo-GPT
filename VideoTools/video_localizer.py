from ImageTools.visual_question_answering import VisualQuestionAnswering
from VideoTools.vid2frames import Video2Frames
from ImageTools.imgutils import prompts
from pathlib import Path
import os
import openai
from sentence_transformers import SentenceTransformer, util
import torch


class SimpleVideoLocalizer:
    template_model = True
    def __init__(self, Video2Frames: Video2Frames, VisualQuestionAnswering: VisualQuestionAnswering):
        print("Initializing SimpleVideoLocalizer")
        self.frames_extractor = Video2Frames
        self.image_qa = VisualQuestionAnswering
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    @prompts(name="Video Localizer",
             description="This tool is used to get the localized moments in the video. The "
                         "input to this tool should be a comma separated string of two, "
                         "representing the video path and the information about the moment."
                         "It works by asking an image QA model whether the frame contains the "
                         "given information. Information should be in descriptive format."
                         "The output is the start and end second of that moment.")
    def inference(self, inputs):
        video_path, information = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        uid = Path(video_path).stem

        frame_dir = os.path.join('video', 'frames', uid)
        if not os.path.exists(frame_dir):
            self.frames_extractor.inference(video_path)

        full_prompt = "Does the image contain this information: " + information
        describe = "The image picture of "

        saved_frames = []
        frame_count = 0
        frame_skip = 5

        for filename in sorted(os.listdir(frame_dir)):
            if filename.endswith(".jpg"):
                frame_path = os.path.join(frame_dir, filename)
                inf_input = frame_path + "," + full_prompt
                if frame_count % frame_skip == 0:
                    description = self.image_qa.inference(describe)
                    result = self.image_qa.inference(inf_input)[:3].lower()
                    if result == "yes" and self.cosine_similarity(description, information) >= 0.65:
                        saved_frames.append(frame_path)
                frame_count += 1

        if len(saved_frames) == 0:
            return "The video contains no given information."
        start = frame_name_to_seconds(frame_name=saved_frames[0])
        end = frame_name_to_seconds(frame_name=saved_frames[len(saved_frames) - 1])
        return start, end

    def cosine_similarity(self, sentence1, sentence2):
        # Encode the sentences to get their embeddings
        embedding1 = self.semantic_model.encode(sentence1, convert_to_tensor=True)
        embedding2 = self.semantic_model.encode(sentence2, convert_to_tensor=True)

        # Compute cosine similarity
        return util.pytorch_cos_sim(embedding1, embedding2)





def frame_name_to_seconds(frame_name, fps=30):
    frame_number_str = frame_name.split("_")[-1].split(".")[0]
    frame_number = int(frame_number_str)

    seconds = frame_number / fps

    return seconds


