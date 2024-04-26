from ImageTools.visual_question_answering import VisualQuestionAnswering
from VideoTools.vid2frames import Video2Frames
from ImageTools.imgutils import prompts
from pathlib import Path
import os


class SimpleVideoLocalizer:
    template_model = True
    def __init__(self, Video2Frames: Video2Frames, VisualQuestionAnswering: VisualQuestionAnswering):
        print("Initializing SimpleVideoLocalizer")
        self.frames_extractor = Video2Frames
        self.image_qa = VisualQuestionAnswering

    @prompts(name="Video Localizer",
             description="This tool is used to get the localized moments in the video. The "
                         "input to this tool should be a comma separated string of two, "
                         "representing the video path and the information about the moment."
                         "It works by comparing the information and the each frame of the video,"
                         "whether the frame contains the given information. So the information"
                         "format should be descriptive. The output is the start and end second of "
                         "that moment.")
    def inference(self, inputs):
        video_path, information = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        uid = Path(video_path).stem

        frame_dir = os.path.join('video', 'frames', uid)
        if not os.path.exists(frame_dir):
            self.frames_extractor.inference(video_path)

        loc_prompt = "Does the frame accurately contain the necessary details for this information:"

        full_prompt = loc_prompt + " " + information + ". Answer only with yes or no:"

        saved_frames = []
        frame_count = 0  # Ensure frame_count is initialized outside the loop
        frame_skip = 5  # Adjust frame_skip to your desired interval

        for filename in sorted(os.listdir(frame_dir)):
            if filename.endswith(".jpg"):
                frame_path = os.path.join(frame_dir, filename)
                inf_input = frame_path + "," + full_prompt
                if frame_count % frame_skip == 0:
                    result = self.image_qa.inference(inf_input)[:3].lower()
                    if result == "yes":
                        saved_frames.append(frame_path)
                frame_count += 1

        if len(saved_frames) == 0:
            raise ValueError("The video contains no given information.")
        start = frame_name_to_seconds(frame_name=saved_frames[0])
        end = frame_name_to_seconds(frame_name=saved_frames[len(saved_frames)])
        return start, end


def frame_name_to_seconds(frame_name, fps=30):
    frame_number_str = frame_name.split("_")[-1].split(".")[0]
    frame_number = int(frame_number_str)

    seconds = frame_number / fps

    return seconds
