import os
import random
import time
import uuid

import requests
from ImageTools.imgutils import prompts

class Video2Video:
    def __init__(self, device):
        self.url_endpoint = "https://hjx99ogdyc2r9x-5000.proxy.runpod.net/api/edit_video"
        self.status_endpoint = "https://hjx99ogdyc2r9x-5000.proxy.runpod.net/api/status/"
        self.download_endpoint = "https://hjx99ogdyc2r9x-5000.proxy.runpod.net/api/download/"
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
        form_data = {'prompt': prompt}

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
