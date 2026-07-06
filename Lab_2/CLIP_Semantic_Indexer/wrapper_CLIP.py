from transformers import CLIPProcessor, CLIPModel
import torch
import os

class wrapper_CLIP:
    def __init__(self, model_name: str, model_save_path: str = "./clip_model"):
        if not os.path.exists(model_save_path):
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
            self.model.save_pretrained(model_save_path)
            self.processor.save_pretrained(model_save_path)
        else:
            self.model = CLIPModel.from_pretrained(model_save_path)
            self.processor = CLIPProcessor.from_pretrained(model_save_path, use_fast=False)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device) # type: ignore

    def _encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device) # type: ignore
        outputs = self.model.get_text_features(**inputs).pooler_output # type: ignore
        return outputs

    def _encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device) # type: ignore
        outputs = self.model.get_image_features(**inputs).pooler_output # type: ignore
        return outputs
    
    def encode(self, text=None, image_batch=None):
            with torch.no_grad():   
                if text is not None:
                    text_features = self._encode_text(text)
                    return text_features
                elif image_batch is not None:
                    image_features = self._encode_image(image_batch)
                    return image_features
                else:
                    raise ValueError("Either text or image must be provided for encoding.")