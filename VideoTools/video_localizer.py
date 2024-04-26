from ImageTools.visual_question_answering import VisualQuestionAnswering
from VideoTools.vid2frames import Video2Frames
from ImageTools.imgutils import prompts
from ImageTools.image_captioning import ImageCaptioning
from pathlib import Path
import os
import openai
from sentence_transformers import SentenceTransformer, util
import torch


class SimpleVideoLocalizer:
    template_model = True
    def __init__(self, ImageCaptioning: ImageCaptioning, Video2Frames: Video2Frames, VisualQuestionAnswering: VisualQuestionAnswering):
        print("Initializing SimpleVideoLocalizer")
        self.frames_extractor = Video2Frames
        self.image_qa = VisualQuestionAnswering
        self.image_capt =ImageCaptioning
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    @prompts(name="Video Localizer",
             description="This tool is used to get the localized moments in the video. The "
                         "input to this tool should be a comma separated string of three, "
                         "representing the video path, the question, and descriptive format"
                         "of that question (information)."
                         "It works by asking Image QA model always in this format 'Is there ... in the image?'"
                         "For example: question = 'is there a man walking in the image?' and"
                         "the information = 'a man walking'."
                         "The output is the start and end second of that moment.")
    def inference(self, inputs):
        # Assume the input format is "video_path,question,information"
        parts = inputs.split(',', 2)  # Splits into at most three parts
        video_path = parts[0]
        question = parts[1] if len(parts) > 1 else None  # Safe access
        information = parts[2] if len(parts) > 2 else None  # Safe access

        uid = Path(video_path).stem

        frame_dir = os.path.join('video', 'frames', uid)
        if not os.path.exists(frame_dir):
            self.frames_extractor.inference(video_path)

        full_prompt = question

        saved_frames = []
        frame_count = 0
        frame_skip = 5

        for filename in sorted(os.listdir(frame_dir)):
            if filename.endswith(".jpg"):
                frame_path = os.path.join(frame_dir, filename)
                inf_input = frame_path + "," + full_prompt
                if frame_count % frame_skip == 0:
                    description = self.image_capt.inference(frame_path)
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
        print("Sentence 1 : " + sentence1)
        print("Sentence 2 : " + sentence2)
        embedding1 = self.semantic_model.encode(sentence1, convert_to_tensor=True)
        embedding2 = self.semantic_model.encode(sentence2, convert_to_tensor=True)

        sim = util.pytorch_cos_sim(embedding1, embedding2)
        print("Cosine similarity: " + str(sim.item()))
        return sim





def frame_name_to_seconds(frame_name, fps=30):
    frame_number_str = frame_name.split("_")[-1].split(".")[0]
    frame_number = int(frame_number_str)

    seconds = frame_number / fps

    return seconds


