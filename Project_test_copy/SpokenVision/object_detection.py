from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

def load_model():
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')
    return model

def detect_objects(image, model, conf_threshold=0.5):
    """
    Detect objects in an image using YOLOv8
    
    Args:
        image: Can be a file path, PIL Image, or numpy array
        model: YOLO model from load_model()
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Dictionary with boxes, labels, scores
    """
    # Process the image depending on its type
    if isinstance(image, str):
        # No need to preprocess if it's a path - YOLO handles this
        pass
    elif not isinstance(image, (Image.Image, np.ndarray)):
        raise ValueError("Invalid image input. Provide a PIL Image, file path, or numpy array")
    
    # Run inference with YOLO
    results = model(image, conf=conf_threshold, verbose=False)
    
    # Process results into a similar format as you were using before
    boxes = []
    labels = []
    scores = []
    
    # Extract detection information
    for r in results:
        for box in r.boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert to list
            boxes.append([x1, y1, x2, y2])
            
            # Get class and confidence
            cls = int(box.cls[0])
            labels.append(r.names[cls])
            scores.append(float(box.conf[0]))
    
    return {
        "boxes": boxes,
        "labels": labels,
        "scores": scores
    }