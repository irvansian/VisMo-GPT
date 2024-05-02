import cv2
import os
from pathlib import Path

from ImageTools.imgutils import prompts

class Video2Frames:
    def __init__(self, device):
        print("Initializing Video2Frames")

    @prompts(name="Extract Video Frames",
             description="useful when you want to extract the frames of a video "
                         "The input to this tool should be a video_path,"
                         "The output of this tool is the dir where the "
                         "frames are saved.")
    def inference(self, video_path, start_second=None, end_second=None):
    # Check if video exists
        if not os.path.exists(video_path):
            print(f"Error: Video path does not exist {video_path}")
            return

    # Create a VideoCapture object
        cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

    # Set the starting frame
        if start_second is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_second * 1000)

    # Get video UID from the path
        uid = Path(video_path).stem  # Extracts uid from file name 'video/{uid}.mp4'

    # Create directory for frames relative to the current script location
        frames_dir = os.path.join('video', 'frames', uid)
        os.makedirs(frames_dir, exist_ok=True)

    # Read until video is completed or end_second is reached
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()

        # If frame is read correctly, ret is True
            if not ret:
                break

        # If end_second is defined and we have reached or passed it, break the loop
            current_second = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if end_second is not None and current_second > end_second:
                break

        # Save frame as image file
            frame_path = os.path.join(frames_dir, f'frame_{frame_id:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_id += 1

    # When everything done, release the video capture object
        cap.release()
        print(f"Frames extracted to directory: {frames_dir}")
        return frames_dir

