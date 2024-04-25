from ImageTools.image_captioning import ImageCaptioning
from vid2frames import Video2Frames

class VideoCaptioning:
    def __init__(self, device):
        self.device = device
        self.imageCaptioning = ImageCaptioning(device=device)

    def inference(self, video_path):

