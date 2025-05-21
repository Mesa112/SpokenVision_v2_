from PIL import Image
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2

blip_processor = None

def load_blip_captioning_model():
    global blip_processor
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model.eval() 
    return blip_model

def generate_caption(frame, blip_model):
    pil_image = Image.fromarray(frame)
    inputs = blip_processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        output = blip_model.generate(**inputs)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)

    return caption