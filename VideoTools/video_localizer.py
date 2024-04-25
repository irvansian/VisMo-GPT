from ImageTools.visual_question_answering import VisualQuestionAnswering
from VideoTolls.vid2frames import Video2Frames
from ImageTools.imgutils import prompts
from pathlib import Path
import os


class SimpleVideoLocalizer:
    def __init__(self, device):
        print("Initializing SimpleVideoLocalizer")
        self.frames_extractor = Video2Frames(device=device)
        self.image_qa = VisualQuestionAnswering(device=device)

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

        frame_dir = '/video/frames/' + uid

        # extract frame first if not done
        if not os.path.exists(frame_dir):
            self.frames_extractor.inference(video_path)

        loc_prompt = "Does the frame accurately contain the necessary details for this information:"

        full_prompt = loc_prompt + " " + information + ". Answer only with yes or no:"

        saved_frames = []
        frame_skip = 30
        frame_count = 0

        for filename in os.listdir(frame_dir):
            if filename.endswith(".jpg"):
                frame_path = os.path.join(frame_dir, filename)
                inputs = frame_path + "," + full_prompt
                if frame_count % frame_skip == 0 and self.image_qa.inference(inputs) == 'yes':
                    saved_frames.append(frame_path)
                frame_count += 1

        start = frame_name_to_seconds(frame_name=saved_frames[0])
        end = frame_name_to_seconds(frame_name=saved_frames[len(saved_frames)])
        return start, end


def frame_name_to_seconds(frame_name, fps=30):
    frame_number_str = frame_name.split("_")[-1].split(".")[0]
    frame_number = int(frame_number_str)

    seconds = frame_number / fps

    return seconds
