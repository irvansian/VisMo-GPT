import cv2
import os

from ImageTools.imgutils import prompts

class Video2Frames:
    def __init__(self, device):
        print("Initializing Video2Frames")

    @prompts(name="Extract Video Frames",
             description="useful when you want to extract the frames of a video "
                         "The input to this tool should be a video_path, start_second "
                         "(default value is start of the video), end_second (default value is "
                         "end of the video).")
    def inference(self, video_path, start_second=None, end_second=None):
        filename_with_extension = video_path.split('/')[-1]
        print(filename_with_extension)
        filename_without_extension = filename_with_extension.split('.')[0]
        output_dir = os.path.join('video', 'frames', filename_without_extension)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            cap.release()
            return None 

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        # Set default start and end times if they are not provided
        start_second = start_second if start_second is not None else 0
        end_second = end_second if end_second is not None else video_duration

        # Calculate start and end frames
        start_frame = int(start_second * fps)
        end_frame = int(end_second * fps)

        # Set the initial position of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output_frame_number = 0
        frame_number = start_frame
        while cap.isOpened() and frame_number <= end_frame:
            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(output_dir, f'{output_frame_number:03d}.jpg')
                cv2.imwrite(frame_filename, frame)
                frame_number += 1
                output_frame_number += 1
            else:
                break

        cap.release()
        return output_dir  # Return the frames per second
