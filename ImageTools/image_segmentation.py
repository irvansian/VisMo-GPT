import os

import cv2
import numpy as np
import torch
import random
import gradio as gr

import wget
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from matplotlib import pyplot as plt
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

from ImageTools.imgutils import prompts, seed_everything, get_new_image_name
from ImageTools.image_boxing import Text2Box


class SegText2Image:
    def __init__(self, device):
        print(f"Initializing SegText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-seg",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker'),
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                            ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Segmentations",
             description="useful when you want to generate a new real image from both the user description and segmentations. "
                         "like: generate a real image of a object or something from this segmentation image, "
                         "or generate a new real image of a object or something from these segmentations. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{instruct_text}, {self.a_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        image.save(updated_image_path)
        print(f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Segmenting:
    def __init__(self, device):
        print(f"Inintializing Segmentation to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_checkpoint_path = os.path.join("checkpoints", "sam")

        self.download_parameters()
        self.sam = build_sam(checkpoint=self.model_checkpoint_path).to(device)
        self.sam_predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        self.saved_points = []
        self.saved_labels = []

    def download_parameters(self):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url, out=self.model_checkpoint_path)

    def show_mask(self, mask: np.ndarray, image: np.ndarray,
                  random_color: bool = False, transparency=1) -> np.ndarray:

        """Visualize a mask on top of an image.
        Args:
            mask (np.ndarray): A 2D array of shape (H, W).
            image (np.ndarray): A 3D array of shape (H, W, 3).
            random_color (bool): Whether to use a random color for the mask.
        Outputs:
            np.ndarray: A 3D array of shape (H, W, 3) with the mask
            visualized on top of the image.
            transparenccy: the transparency of the segmentation mask
        """

        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

        image = cv2.addWeighted(image, 0.7, mask_image.astype('uint8'), transparency, 0)

        return image

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)

    def get_mask_with_boxes(self, image_pil, image, boxes_filt):

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        return masks

    def segment_image_with_boxes(self, image_pil, image_path, boxes_filt, pred_phrases):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image)

        masks = self.get_mask_with_boxes(image_pil, image, boxes_filt)

        # draw output image

        for mask in masks:
            image = self.show_mask(mask[0].cpu().numpy(), image, random_color=True, transparency=0.3)

        updated_image_path = get_new_image_name(image_path, func_name="segmentation")

        new_image = Image.fromarray(image)
        new_image.save(updated_image_path)

        return updated_image_path

    def set_image(self, img) -> None:
        """Set the image for the predictor."""
        with torch.cuda.amp.autocast():
            self.sam_predictor.set_image(img)

    def show_points(self, coords: np.ndarray, labels: np.ndarray,
                    image: np.ndarray) -> np.ndarray:
        """Visualize points on top of an image.

        Args:
            coords (np.ndarray): A 2D array of shape (N, 2).
            labels (np.ndarray): A 1D array of shape (N,).
            image (np.ndarray): A 3D array of shape (H, W, 3).
        Returns:
            np.ndarray: A 3D array of shape (H, W, 3) with the points
            visualized on top of the image.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        for p in pos_points:
            image = cv2.circle(
                image, p.astype(int), radius=3, color=(0, 255, 0), thickness=-1)
        for p in neg_points:
            image = cv2.circle(
                image, p.astype(int), radius=3, color=(255, 0, 0), thickness=-1)
        return image

    def segment_image_with_click(self, img, is_positive: bool,
                                 evt: gr.SelectData):

        self.sam_predictor.set_image(img)
        self.saved_points.append([evt.index[0], evt.index[1]])
        self.saved_labels.append(1 if is_positive else 0)
        input_point = np.array(self.saved_points)
        input_label = np.array(self.saved_labels)

        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

        img = self.show_mask(masks[0], img, random_color=False, transparency=0.3)

        img = self.show_points(input_point, input_label, img)

        return img

    def segment_image_with_coordinate(self, img, is_positive: bool,
                                      coordinate: tuple):
        '''
            Args:
                img (numpy.ndarray): the given image, shape: H x W x 3.
                is_positive: whether the click is positive, if want to add mask use True else False.
                coordinate: the position of the click
                          If the position is (x,y), means click at the x-th column and y-th row of the pixel matrix.
                          So x correspond to W, and y correspond to H.
            Output:
                img (PLI.Image.Image): the result image
                result_mask (numpy.ndarray): the result mask, shape: H x W

            Other parameters:
                transparency (float): the transparenccy of the mask
                                      to control he degree of transparency after the mask is superimposed.
                                      if transparency=1, then the masked part will be completely replaced with other colors.
        '''
        self.sam_predictor.set_image(img)
        self.saved_points.append([coordinate[0], coordinate[1]])
        self.saved_labels.append(1 if is_positive else 0)
        input_point = np.array(self.saved_points)
        input_label = np.array(self.saved_labels)

        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

        img = self.show_mask(masks[0], img, random_color=False, transparency=0.3)

        img = self.show_points(input_point, input_label, img)

        img = Image.fromarray(img)

        result_mask = masks[0]

        return img, result_mask

    @prompts(name="Segment the Image",
             description="useful when you want to segment all the part of the image, but not segment a certain object."
                         "like: segment all the object in this image, or generate segmentations on this image, "
                         "or segment the image,"
                         "or perform segmentation on this image, "
                         "or segment all the object in this image."
                         "The input to this tool should be a string, representing the image_path")
    def inference_all(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        if len(masks) == 0:
            return
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m)))

        updated_image_path = get_new_image_name(image_path, func_name="segment-image")
        plt.axis('off')
        plt.savefig(
            updated_image_path,
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        return updated_image_path


class ObjectSegmenting:
    template_model = True  # Add this line to show this is a template model.

    def __init__(self, Text2Box: Text2Box, Segmenting: Segmenting):
        # self.llm = OpenAI(temperature=0)
        self.grounding = Text2Box
        self.sam = Segmenting

    @prompts(name="Segment the given object",
             description="useful when you only want to segment the certain objects in the picture"
                         "according to the given text"
                         "like: segment the cat,"
                         "or can you segment an obeject for me"
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be found")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        print(f"image_path={image_path}, text_prompt={det_prompt}")
        image_pil, image = self.grounding.load_image(image_path)

        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(image, det_prompt)
        updated_image_path = self.sam.segment_image_with_boxes(image_pil, image_path, boxes_filt, pred_phrases)
        print(
            f"\nProcessed ObejectSegmenting, Input Image: {image_path}, Object to be Segment {det_prompt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path

    def merge_masks(self, masks):
        '''
            Args:
                mask (numpy.ndarray): shape N x 1 x H x W
            Outputs:
                new_mask (numpy.ndarray): shape H x W
        '''
        if type(masks) == torch.Tensor:
            x = masks
        elif type(masks) == np.ndarray:
            x = torch.tensor(masks, dtype=int)
        else:
            raise TypeError("the type of the input masks must be numpy.ndarray or torch.tensor")
        x = x.squeeze(dim=1)
        value, _ = x.max(dim=0)
        new_mask = value.cpu().numpy()
        new_mask.astype(np.uint8)
        return new_mask

    def get_mask(self, image_path, text_prompt):

        print(f"image_path={image_path}, text_prompt={text_prompt}")
        # image_pil (PIL.Image.Image) -> size: W x H
        # image (numpy.ndarray) -> H x W x 3
        image_pil, image = self.grounding.load_image(image_path)

        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(image, text_prompt)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam.sam_predictor.set_image(image)

        # masks (torch.tensor) -> N x 1 x H x W
        masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)

        # merged_mask -> H x W
        merged_mask = self.merge_masks(masks)
        # draw output image

        for mask in masks:
            image = self.sam.show_mask(mask[0].cpu().numpy(), image, random_color=True, transparency=0.3)

        merged_mask_image = Image.fromarray(merged_mask)

        return merged_mask