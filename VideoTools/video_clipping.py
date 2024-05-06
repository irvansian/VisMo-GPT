import uuid

import cv2
import os

import numpy as np
from ImageTools.imgutils import prompts


class VideoClipping():
    def __init__(self, device):
        print(f"Initializing VideoClipping")

    @prompts(name="Extract Clip Video",
             description="useful when you want to get the subvideo from a certain second to a certain second. "
                         "like: clip/cut the video from the 5th second to the 10th second. "
                         "The input to this tool should be a comma separated string of 3, "
                         "representing the video path, start_time (start second), and end_time (end second). ")
    def inference_extract_subvideo(self, inputs):
        input_video, start_time, end_time = inputs.split(",")[0], float(inputs.split(",")[1]), float(inputs.split(",")[2])
        output_video = os.path.join('video', f"{str(uuid.uuid4())[:8]}.mp4")
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"Failed to open video: {input_video}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if start_frame <= frame_number <= end_frame:
                    out.write(frame)
                frame_number += 1
                if frame_number > end_frame:
                    break
            else:
                break

        # Release the video objects
        cap.release()
        out.release()

        print(f"Subvideo saved to: {output_video}")
        return output_video

    def extract_frames(self, video_path, start_second, end_second, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            cap.release()
            return None  # Return None if video file could not be opened
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        start_frame = start_second * fps
        end_frame = end_second * fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output_frame_number = 0
        frame_number = start_frame
        while cap.isOpened() and frame_number <= end_frame:
            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(output_dir, f'{output_frame_number:04d}.png')
                cv2.imwrite(frame_filename, frame)
                frame_number += 1
                output_frame_number += 1
            else:
                break
        cap.release()
        return fps  # Return the frames per second
    def stitch(self, inputs):
        list_video_paths = inputs.split(',')
        video_captures = []
        frame_rates = []

        for path in list_video_paths:
            try:
                cap = cv2.VideoCapture(path.strip())
                if not cap.isOpened():
                    print(f"Error opening video: {path}")
                    return None
                # Retrieve and store the frame rate
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_rates.append(fps)
                video_captures.append(cap)
            except Exception as e:
                print(f"Error processing video {path}: {e}")
                return None

        average_fps = np.mean(frame_rates) if frame_rates else 24

        # Determine the frame size from the first video
        ret, frame = video_captures[0].read()
        if not ret:
            print("Error reading the first frame.")
            for cap in video_captures:
                cap.release()
            return None
        first_video_width = frame.shape[1]
        first_video_height = frame.shape[0]

        video_filename = os.path.join('video', f"{str(uuid.uuid4())[:8]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, average_fps, (first_video_width, first_video_height))

        # Rewind the first video capture to start from the beginning
        video_captures[0].set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            frames = []
            all_frames_read = True

            for cap in video_captures:
                ret, frame = cap.read()
                if ret:
                    frame = self.resize_frame_to_match_first(frame, first_video_width, first_video_height)
                    frames.append(frame)
                else:
                    all_frames_read = False
                    break

            if not all_frames_read:
                break

            stitched_frame = cv2.hconcat(frames)
            out.write(stitched_frame)

        for cap in video_captures:
            cap.release()
        out.release()

        return video_filename

    def resize_frame_to_match_first(self, frame, width, height):
        return cv2.resize(frame, (width, height))




