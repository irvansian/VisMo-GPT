import os
import platform
import subprocess
import uuid

import cv2
import numpy as np
import torch

from image_generation import createImage

from ImageTools.image_boxing import Text2Box
from ImageTools.image_segmentation import Segmenting
from ImageTools.image_inpainting import Inpainting

import vidutils

class Video2Video:
    template_model=True
    def __init__(self, Text2Box: Text2Box, Segmenting: Segmenting, Inpainting: Inpainting):
        print("Initializing Text2Video")

    def inference(self, inputs):
        '''
        1. Generate bbox / segment
        2. Create mask
        3. Inpaint the mask
        4. Propagate edit with ebsynth
        '''

        video_path = inputs.split(",")[0]
        start_time = float(inputs.split(",")[1])
        end_time = float(inputs.split(",")[2])
        prompt = inputs.split(",")[3]

        frames = self.extract_frames(video_path, 5)
        grid = self.create_grid(frames)
        edited_grid = createImage(grid, prompt)
        edited_frames = self.split_grid(edited_grid)

    def extract_frames(self, video_path, duration):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frames = []

        # Extract the first frame
        ret, first_frame = cap.read()
        frames.append(first_frame)

        # Analyze middle frames for object movement
        last_frame = first_frame
        diffs = []
        for i in range(1, min(duration * fps, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            diff = cv2.absdiff(frame, last_frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            score = np.sum(gray)
            diffs.append((score, frame))
            last_frame = frame

        # Extract the two frames with the highest movement score
        diffs.sort(key=lambda x: -x[0])
        frames.append(diffs[0][1])
        if len(diffs) > 1:
            frames.append(diffs[1][1])

        # Extract the last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        frames.append(last_frame)

        cap.release()
        return frames

    def create_grid(self, frames):
        """Combine 4 frames into a 2x2 grid."""
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        grid = np.vstack((top_row, bottom_row))
        return grid


    def split_grid(self, grid):
        """Split a 2x2 grid image into its 4 original frames."""
        h, w, _ = grid.shape
        mid_h, mid_w = h // 2, w // 2

        frame1 = grid[:mid_h, :mid_w]
        frame2 = grid[:mid_h, mid_w:]
        frame3 = grid[mid_h:, :mid_w]
        frame4 = grid[mid_h:, mid_w:]

        return [frame1, frame2, frame3, frame4]

class VideoStylization:
    def __init__(self, device):
        print("Initializing VideoStylization")
        os_name = platform.system()
        cuda_available = torch.cuda.is_available()
        self.ebsynth_bin = None
        if os_name == "Windows":
            self.ebsynth_bin = os.path.join('ebsynth', "ebsynth-windows-cpu")
        elif os_name == "Darwin":
            self.ebsynth_bin = os.path.join('ebsynth', "ebsynth-macos-cpu")
        elif os_name == "Linux":
            self.ebsynth_bin = os.path.join('ebsynth', "ebsynth-linux-cpu")
        else:
            return "Unknown OS"

        if (cuda_available):
            self.ebsynth_bin = self.ebsynth_bin + "+cuda"
        print("ebsynth : " + self.ebsynth_bin)

    def inference(self, image_reference_path, frames):
        styled_image_paths = []
        for index, frame in enumerate(frames):
            styled_image_paths.append(
                self.run_image_on_ebsynth(image_reference_path, os.path.join("video", "frames_output", "disas", "000.jpg"), frame, index))

        # Assuming all images are of the same size, read the first image to get the dimensions
        frame = cv2.imread(styled_image_paths[0])
        h, w, layers = frame.shape
        size = (w, h)

        # Define the codec and create a VideoWriter object
        video_path = os.path.join('..', "video", f"{str(uuid.uuid4())[:8]}.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              size)  # 30 is the FPS. Adjust as needed.

        for i in range(len(styled_image_paths)):
            img = cv2.imread(styled_image_paths[i])
            out.write(img)

        out.release()
        return video_path

    def run_image_on_ebsynth(self, style_path, input_path, target_path, index):
        print("lewat sini")
        output_directory = os.path.join("..", "video", "tempimages")
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, f"{index}.png")

        cmd = [
            self.ebsynth_bin,
            '-style', style_path,
            '-guide', input_path, target_path,
            '-weight', '2',
            '-output', output_path
        ]
        print(cmd)
        subprocess.run(cmd)
        return output_path

def get_all_images(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    images = [file for file in files if os.path.splitext(file)[1].lower() in extensions]
    return images

if __name__ == "__main__":
    # video_path = os.path.join("..", "video", "disas.mp4")
    style_image_path = os.path.join("..", "image", "style.jpg")
    # vidutils.extract_frames(video_path, 3, 8, os.path.join("video", "frames_output", "disas"))
    video_styler = VideoStylization('cpu')
    video_frames_path = os.path.join("video", "frames_output", "disas")
    result = video_styler.inference(style_image_path, get_all_images(video_frames_path))
    # print(result)
    styled_frame_path = video_path = os.path.join("..", "video", "tempimages")
    result_vid = os.path.join("..", "video", "disas_styled2.mp4")
    vidutils.frames_to_video(styled_frame_path, result_vid, 30)

