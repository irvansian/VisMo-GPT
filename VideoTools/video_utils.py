from moviepy.video.io.VideoFileClip import VideoFileClip
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

        # Specify the directory to save the video and the new filename
        output_path = 'video'
        filename = f'{uid}.mp4'

        # Pass the output_path and filename to the download function
        stream.download(output_path=output_path, filename=filename)

        # Build the full destination path
        dest_path = os.path.join(output_path, filename)

        print(f"Downloaded '{yt.title}' successfully to {dest_path}.")
        return dest_path

class VideoMetaData:
    def __init__(self, device):
        print("Initializing Video Data")

    @prompts(name="Get Video Metadata", description="Tool to get video metadata. The input is video path.")
    def inference(self, video_path):
        try:
            with VideoFileClip(video_path) as clip:
                width, height = clip.size
                duration = clip.duration
                fps = clip.fps

            metadata = {
                'Width': width,
                'Height': height,
                'Duration (seconds)': duration,
                'Frames per second (FPS)': fps
            }

            return metadata
        except Exception as e:
            print(f"Error occurred: {e}")
            return {}


