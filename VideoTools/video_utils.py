from pytube import YouTube
import uuid
import os
from ImageTools.imgutils import prompts
class VideoDownload:
    def __init__(self, device):
        print('Initializing Video Downloader')

    @prompts(name="Youtube Video Downloader",
             description="Use this tool to download video from youtube. The input is"
                         "the youtube url link to download. Output is the video path.")
    def inference(self, url):
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first()
        uid = uuid.uuid4().hex[:8]

        if not os.path.exists('video'):
            os.makedirs('video')

        dest_path = os.path.join('video', f'{uid}.mp4')

        stream.download(dest_path)
        print(f"Downloaded '{yt.title}' successfully.")
        return dest_path
