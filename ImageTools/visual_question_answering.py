import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from ImageTools.imgutils import prompts
from PIL import Image


class VisualQuestionAnswering:
    def __init__(self, device):
        # print(f"Initializing VisualQuestionAnswering to {device}")
        # self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        # self.device = device
        # self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        # self.model = BlipForQuestionAnswering.from_pretrained(
        #     "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype).to(self.device)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.model.to(self.device)

    @prompts(name="Answer Question About The Image",
             description="useful when you need an answer for a question based on an image. "
                         "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the question")
    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert('RGB')
        # inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        # out = self.model.generate(**inputs)
        # answer = self.processor.decode(out[0], skip_special_tokens=True)
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(answer)
        print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
              f"Output Answer: {answer}")
        return answer