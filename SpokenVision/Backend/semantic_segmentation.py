from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch

def load_model():
    """
    Load the SegFormer segmentation model and processor.
    Returns:
    model: The segmentation model
    processor: Image processor for preprocessing
    """
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    
    processor = SegformerImageProcessor.from_pretrained(model_name, use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    
    return model, processor

def preprocess_image(image_path, processor):
    """
    Preprocess the image for the segmentation model.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def predict_segmentation(image_path, model, processor):
    """
    Generate a segmentation map from an input image.
    """
    inputs = preprocess_image(image_path, processor)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, num_classes, H, W)

    segmentation_map = logits.argmax(dim=1)[0]
    return segmentation_map

def visualize_segmentation(image_path, segmentation_map):
    """
    Visualize the segmentation result.
    """
    image = Image.open(image_path)
    segmentation_map_np = segmentation_map.byte().cpu().numpy()
    
    seg_img = Image.fromarray(segmentation_map_np)
    seg_img.putpalette([i for _ in range(256) for i in range(3)])  # simple grayscale palette
    image.show()
    seg_img.show()