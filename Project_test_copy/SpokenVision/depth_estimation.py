from PIL import Image
import numpy as np
import torch
import cv2

from transformers import DPTImageProcessor, DPTForDepthEstimation

image_processor = None

def load_depth_model():
    global image_processor
    image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas", use_fast=True)
    midas_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
    midas_model.eval()  # Set the model to evaluation mode
    return midas_model


def preprocess_frame(frame):
    pil_image = Image.fromarray(frame)
    inputs = image_processor(pil_image, return_tensors="pt")

    return inputs

def estimate_depth(frame, midas_model):
    inputs = preprocess_frame(frame)

    with torch.no_grad():
        depth_map = midas_model(**inputs).predicted_depth

    depth_map = depth_map.squeeze().cpu().numpy()

    # normalize the depth map to [0, 255] range for visualization
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_map = depth_map.astype(np.uint8)

    # Resize depth map to match the original image's dimensions
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    return depth_map