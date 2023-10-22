import math

import torch
from PIL import ImageOps, Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from langchain.llms.openai import OpenAI

from ImageTools.imgutils import blend_gt2pt, prompts, get_new_image_name


class Inpainting:
    def __init__(self, device):
        self.device = device
        self.revision = 'fp16' if 'cuda' in self.device else None
        self.torch_dtype = torch.float16 if 'cuda' in self.device else torch.float32

        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype,safety_checker=StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker')).to(device)
    def __call__(self, prompt, image, mask_image, height=512, width=512, num_inference_steps=50):
        update_image = self.inpaint(prompt=prompt, image=image.resize((width, height)),
                                     mask_image=mask_image.resize((width, height)), height=height, width=width, num_inference_steps=num_inference_steps).images[0]
        return update_image

class InfinityOutPainting:
    template_model = True # Add this line to show this is a template model.
    def __init__(self, ImageCaptioning, Inpainting, VisualQuestionAnswering):
        self.llm = OpenAI(temperature=0)
        self.ImageCaption = ImageCaptioning
        self.inpaint = Inpainting
        self.ImageVQA = VisualQuestionAnswering
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    def get_BLIP_vqa(self, image, question):
        inputs = self.ImageVQA.processor(image, question, return_tensors="pt").to(self.ImageVQA.device,
                                                                                  self.ImageVQA.torch_dtype)
        out = self.ImageVQA.model.generate(**inputs)
        answer = self.ImageVQA.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed VisualQuestionAnswering, Input Question: {question}, Output Answer: {answer}")
        return answer

    def get_BLIP_caption(self, image):
        inputs = self.ImageCaption.processor(image, return_tensors="pt").to(self.ImageCaption.device,
                                                                                self.ImageCaption.torch_dtype)
        out = self.ImageCaption.model.generate(**inputs)
        BLIP_caption = self.ImageCaption.processor.decode(out[0], skip_special_tokens=True)
        return BLIP_caption

    def check_prompt(self, prompt):
        check = f"Here is a paragraph with adjectives. " \
                f"{prompt} " \
                f"Please change all plural forms in the adjectives to singular forms. "
        return self.llm(check)

    def get_imagine_caption(self, image, imagine):
        BLIP_caption = self.get_BLIP_caption(image)
        background_color = self.get_BLIP_vqa(image, 'what is the background color of this image')
        style = self.get_BLIP_vqa(image, 'what is the style of this image')
        imagine_prompt = f"let's pretend you are an excellent painter and now " \
                         f"there is an incomplete painting with {BLIP_caption} in the center, " \
                         f"please imagine the complete painting and describe it" \
                         f"you should consider the background color is {background_color}, the style is {style}" \
                         f"You should make the painting as vivid and realistic as possible" \
                         f"You can not use words like painting or picture" \
                         f"and you should use no more than 50 words to describe it"
        caption = self.llm(imagine_prompt) if imagine else BLIP_caption
        caption = self.check_prompt(caption)
        print(f'BLIP observation: {BLIP_caption}, ChatGPT imagine to {caption}') if imagine else print(
            f'Prompt: {caption}')
        return caption

    def resize_image(self, image, max_size=1000000, multiple=8):
        aspect_ratio = image.size[0] / image.size[1]
        new_width = int(math.sqrt(max_size * aspect_ratio))
        new_height = int(new_width / aspect_ratio)
        new_width, new_height = new_width - (new_width % multiple), new_height - (new_height % multiple)
        return image.resize((new_width, new_height))

    def dowhile(self, original_img, tosize, expand_ratio, imagine, usr_prompt):
        old_img = original_img
        while (old_img.size != tosize):
            prompt = self.check_prompt(usr_prompt) if usr_prompt else self.get_imagine_caption(old_img, imagine)
            crop_w = 15 if old_img.size[0] != tosize[0] else 0
            crop_h = 15 if old_img.size[1] != tosize[1] else 0
            old_img = ImageOps.crop(old_img, (crop_w, crop_h, crop_w, crop_h))
            temp_canvas_size = (expand_ratio * old_img.width if expand_ratio * old_img.width < tosize[0] else tosize[0],
                                expand_ratio * old_img.height if expand_ratio * old_img.height < tosize[1] else tosize[
                                    1])
            temp_canvas, temp_mask = Image.new("RGB", temp_canvas_size, color="white"), Image.new("L", temp_canvas_size,
                                                                                                  color="white")
            x, y = (temp_canvas.width - old_img.width) // 2, (temp_canvas.height - old_img.height) // 2
            temp_canvas.paste(old_img, (x, y))
            temp_mask.paste(0, (x, y, x + old_img.width, y + old_img.height))
            resized_temp_canvas, resized_temp_mask = self.resize_image(temp_canvas), self.resize_image(temp_mask)
            image = self.inpaint(prompt=prompt, image=resized_temp_canvas, mask_image=resized_temp_mask,
                                              height=resized_temp_canvas.height, width=resized_temp_canvas.width,
                                              num_inference_steps=50).resize(
                (temp_canvas.width, temp_canvas.height), Image.ANTIALIAS)
            image = blend_gt2pt(old_img, image)
            old_img = image
        return old_img

    @prompts(name="Extend An Image",
             description="useful when you need to extend an image into a larger image."
                         "like: extend the image into a resolution of 2048x1024, extend the image into 2048x1024. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the resolution of widthxheight")
    def inference(self, inputs):
        image_path, resolution = inputs.split(',')
        width, height = resolution.split('x')
        tosize = (int(width), int(height))
        image = Image.open(image_path)
        image = ImageOps.crop(image, (10, 10, 10, 10))
        out_painted_image = self.dowhile(image, tosize, 4, True, False)
        updated_image_path = get_new_image_name(image_path, func_name="outpainting")
        out_painted_image.save(updated_image_path)
        print(f"\nProcessed InfinityOutPainting, Input Image: {image_path}, Input Resolution: {resolution}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path