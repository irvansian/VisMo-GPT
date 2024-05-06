import os
import random
import time
import uuid

import requests
from ImageTools.imgutils import prompts
from VideoTools.vid2frames import Video2Frames
from ImageTools.image_captioning import ImageCaptioning

class Video2Video:
    template_model = True
    def __init__(self, ImageCaptioning: ImageCaptioning, Video2Frames: Video2Frames):
        self.url_endpoint = "https://750zn651yz17aq-5000.proxy.runpod.net/api/edit_video"
        self.status_endpoint = "https://750zn651yz17aq-5000.proxy.runpod.net/api/status/"
        self.download_endpoint = "https://750zn651yz17aq-5000.proxy.runpod.net/api/download/"

        self.frame_extractor = Video2Frames
        self.image_captioning = ImageCaptioning
        print("Initializing Text2Video")

    @prompts(name="Edit Video with Natural Language",
             description="useful when you want to edit objects in video with natural language, but "
                         "keeping the movement of the object in video. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the video_path and the prompt for editing the video. "
                         "The prompt should be descriptive and not imperative. If user asks to 'edit "
                         "this video to a girl smiling' the prompt should be 'a girl smiling'.")
    def inference(self, inputs):
        video_path, prompt = inputs.split(",", 1)

        frame_dir = self.frame_extractor.inference(video_path)
        frame_files = [f for f in os.listdir(frame_dir) if f.endswith('.jpg')]  # Adjust the extension if necessary
        frame_files.sort()

        middle_index = len(frame_files) // 2
        middle_frame = frame_files[middle_index] if frame_files else None
        middle_frame_path = os.path.join(frame_dir, middle_frame) if middle_frame else None

        inversion_prompt = self.image_captioning.inference(middle_frame_path)

        print("Middle frame path:", middle_frame_path)
        form_data = {'prompt': prompt, 'inversion_prompt' : inversion_prompt}

        with open(video_path, 'rb') as video_file:
            files = {'video': (video_file.name, video_file, 'video/mp4')}
            response = requests.post(self.url_endpoint, data=form_data, files=files)
            print("response : " + response.text)

        if response.status_code == 200:
            print("Video uploaded successfully.")
            job_id = response.json().get('job_id')
            edited_video_path = self.check_status_and_download(job_id)
            return edited_video_path
        else:
            print(f"Error uploading video. Status code: {response.status_code}, Response text: {response.text}")
            return None

    def check_status_and_download(self, job_id):
        status_endpoint = f"{self.status_endpoint}{job_id}"
        video_path = None
        while True:
            status_response = requests.get(status_endpoint)
            if status_response.status_code == 200:
                status = status_response.json().get('status')
                print(f"Processing status: {status}")
                if status == 'completed':
                    download_link = f"{self.download_endpoint}{job_id}"
                    video_path = self.download_video(download_link)
                    break
                elif status == 'failed':
                    print("Video processing failed.")
                    break
            else:
                print(f"Error checking status. Status code: {status_response.status_code}, Response text: {status_response.text}")
                break
            time.sleep(10)
        return video_path

    def download_video(self, download_link):
        response = requests.get(download_link)
        if response.status_code == 200:
            video_filename = os.path.join('video', f"{str(uuid.uuid4())[:8]}.mp4")

            with open(video_filename, 'wb') as file:
                file.write(response.content)
            print(f"Video downloaded successfully. Saved as {video_filename}.")
            return video_filename
        else:
            print(f"Error downloading video. Status code: {response.status_code}, Response text: {response.text}")
            return None
