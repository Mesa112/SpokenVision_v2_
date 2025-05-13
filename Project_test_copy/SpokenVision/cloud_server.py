from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tempfile
import cv2
import os
import tempfile
from typing import Optional
import base64
import torch.nn.functional as F
import time
import logging
import platform
from pydantic import BaseModel
from anthropic import Anthropic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store models
model = None
segmentation_model = None
feature_extractor = None
depth_model = None
blip_model = None
kokoro_model = None
qwen_model = None
context_builder = None
openai_text_client = None
use_qwen = False

# Lazy loading functions for individual models
def ensure_detection_model():
    global model
    if model is None:
        from object_detection import load_model
        model = load_model()
        logger.info("Object detection model loaded")
    return model

def ensure_segmentation_model():
    global segmentation_model, feature_extractor
    if segmentation_model is None or feature_extractor is None:
        from semantic_segmentation import load_model as load_segmentation_model
        segmentation_model, feature_extractor = load_segmentation_model()
        logger.info("Segmentation model loaded")
    return segmentation_model, feature_extractor

def ensure_depth_model():
    global depth_model
    if depth_model is None:
        from depth_estimation import load_depth_model
        depth_model = load_depth_model()
        logger.info("Depth model loaded")
    return depth_model

def ensure_blip_model():
    global blip_model
    if blip_model is None:
        from blip_image_captioning import load_blip_captioning_model
        blip_model = load_blip_captioning_model()
        logger.info("BLIP captioning model loaded")
    return blip_model

def ensure_kokoro_model():
    global kokoro_model
    if kokoro_model is None:
        from kokoro_audio import load_kokoro_model
        kokoro_model = load_kokoro_model()
        logger.info("Kokoro audio model loaded")
    return kokoro_model

def ensure_context_builder():
    global context_builder
    if context_builder is None:
        from context_builder import ContextBuilder
        context_builder = ContextBuilder()
        logger.info("Context builder initialized")
    return context_builder

def ensure_openai_client():
    global openai_text_client
    if openai_text_client is None:
        from openai import OpenAI
        
        # Directly use the API key instead of loading from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if openai_api_key:
            openai_text_client = OpenAI(
                api_key=openai_api_key,
                timeout=60.0,  # 60 seconds timeout instead of default 10
                max_retries=5   # Increase max retries
            )
            logger.info("OpenAI client initialized")
        else:
            logger.warning("OpenAI API key not found. Caption enhancement will be skipped.")
    return openai_text_client


# Function to improve caption using OpenAI
# Function to improve caption using OpenAI with a retry limit
def improve_caption(raw_caption):
    """Use OpenAI to improve the raw caption."""
    client = ensure_openai_client()
    
    if not client:
        return raw_caption
    
    try:
        # Set a custom timeout for this operation
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        import threading
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Create a thread for the OpenAI call with a timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        lambda: client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that rewrites AI-generated image captions to sound more natural and human-like."},
                                {"role": "user", "content": f"Caption: {raw_caption}"}
                            ],
                            temperature=0.7,
                            max_tokens=100
                        )
                    )
                    # Wait for the response with a timeout
                    response = future.result(timeout=15)  # 15 second timeout
                    return response.choices[0].message.content.strip()
            
            except (TimeoutError, Exception) as e:
                retry_count += 1
                logger.warning(f"OpenAI caption improvement retry {retry_count}/{max_retries} failed: {e}")
                if retry_count >= max_retries:
                    logger.warning(f"Max retries reached for OpenAI, skipping caption enhancement")
                    break
                time.sleep(1)  # Brief pause before retrying
        
        # If we get here, all retries failed, return original caption
        return raw_caption
    
    except Exception as e:
        logger.warning(f"OpenAI caption improvement failed: {e}")
        return raw_caption

# Define the enhance_description_with_gpt function with retry limits
def enhance_description_with_gpt(raw_description):
    """Use GPT-3.5-Turbo to convert technical descriptions into concise, navigational guidance."""
    system_prompt = """
    You are an AI guide helping a blind person navigate. Keep descriptions extremely brief (25 words max).

    Guidelines:
    1. Focus only on the closest, most important objects
    2. Use directions like "in front", "to your left", "ahead"
    3. Mention potential obstacles first
    4. Omit decorative details unless relevant for navigation
    5. Be direct and clear - treat this as real-time guidance
    6. For spatial info, use direction first ("To your right, a table" not "A table to your right")
    7. Be natural like you were talking to your friend
    8. Be fun, dont sound robotic
    9. Use simple language, avoid jargon
    10. Avoid excessive details, focus on immediate surroundings
    11. Use "you" and "your" to make it personal
    12. Use "left" and "right" instead of "to the left" or "to the right"
    13. Avoid "there is" or "there are" - just state the object
    14. Use be careful, watch out, or similar phrases to indicate potential hazards
    
    
    Your output should help someone navigate safely without overwhelming them with details.
    """
    
    client = ensure_openai_client()
    if client:
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            lambda: client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"{raw_description}"}
                                ],
                                max_tokens=50,
                                temperature=0.5
                            )
                        )
                        response = future.result(timeout=15)
                        return response.choices[0].message.content.strip()
                except (TimeoutError, Exception) as e:
                    retry_count += 1
                    logger.warning(f"GPT enhancement retry {retry_count}/{max_retries} failed: {e}")
                    time.sleep(1)
        except Exception as e:
            logger.warning(f"Error using GPT: {e}")

    # Fallback to Claude if OpenAI fails or not configured
    claude_client = ensure_claude_client()
    if claude_client:
        try:
            response = claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": system_prompt + "\n" + raw_description}
                ]
            )
            return response.content[0].text.strip()
        except Exception as ce:
            logger.warning(f"Claude fallback failed: {ce}")

    return raw_description


# Endpoint to process image and audio files
claude_text_client = None

def ensure_claude_client():
    global claude_text_client
    if claude_text_client is None:
        try:
            ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
            claude_api_key = ANTHROPIC_API_KEY
            if claude_api_key:
                claude_text_client = Anthropic(api_key=claude_api_key)
                logger.info("Claude client initialized")
            else:
                logger.warning("Claude API key not found. Fallback will not work.")
        except Exception as e:
            logger.error(f"Error initializing Claude client: {e}")
    return claude_text_client

class TestGptRequest(BaseModel):
    """Request model for testing GPT."""
    text: str
@app.post("/test-gpt/")
async def test_gpt(request: TestGptRequest):
    """Simple endpoint to test if GPT enhancement is working."""
    try:
        client = ensure_openai_client()
        
        if not client:
            logger.warning("OpenAI API key not found during GPT test.")
            return JSONResponse(content={"success": False, "error": "OpenAI API key not found"})
        
        # Try a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with 'GPT test successful' if you can read this."}
            ],
            max_tokens=20
        )
        
        # Check if we got a valid response
        if response and response.choices and len(response.choices) > 0:
            logger.info("GPT test successful")
            return JSONResponse(content={
                "success": True, 
                "message": response.choices[0].message.content
            })
        else:
            logger.warning("GPT test returned empty response")
            return JSONResponse(content={"success": False, "error": "Empty response"})
            
    except Exception as e:
        logger.error(f"Error testing GPT: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)})
@app.post("/process/")
async def process_files(
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None)
):
    logger.info("Received input files.")

    try:
        # Lazy load models inside the route
        detection_model = ensure_detection_model()
        seg_model, feat_extractor = ensure_segmentation_model()
        depth_model = ensure_depth_model()
        blip_model = ensure_blip_model()
        kokoro_model = ensure_kokoro_model()
        context_builder = ensure_context_builder()
        openai_client = ensure_openai_client()

        # Import necessary functions for processing
        from object_detection import detect_objects
        from semantic_segmentation import predict_segmentation
        from depth_estimation import estimate_depth
        from blip_image_captioning import generate_caption
        from kokoro_audio import text_to_audio
        
        # Process the uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]

        # Run object detection
        latest_results = detect_objects(frame, detection_model, conf_threshold=0.3)

        # Run depth estimation
        depth_map = estimate_depth(frame, depth_model)
        latest_depth_map = cv2.resize(depth_map, (frame_width, frame_height))

        logger.info("Object detection and depth estimation completed.")
        
        # Save frame for segmentation
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, frame)
            
        # Run semantic segmentation
        segmentation_map = predict_segmentation(temp_file_path, seg_model, feat_extractor)

        # Resize segmentation map to match frame dimensions
        latest_segmentation_map = F.interpolate(
            segmentation_map.unsqueeze(0).unsqueeze(0).float(),
            size=(frame_height, frame_width),
            mode='nearest'
        ).squeeze().long()

        logger.info("Semantic segmentation completed.")

        # Generate caption
       
        caption = generate_caption(frame, blip_model)

        logger.info("Caption generation completed.")

        # Process context
        raw_context_description = context_builder.process_frame_data(
            latest_results, 
            latest_depth_map, 
            latest_segmentation_map.cpu().numpy(), 
            caption
        )
        logger.info("Context processing completed.")

        # Enhance the description with GPT for natural speech
        caption = enhance_description_with_gpt(raw_context_description)
        logger.info("Caption enhancement completed.")

        # Get the system temp directory 
        temp_dir = tempfile.gettempdir()

        # Create audio output directory
        audio_output_dir = os.path.join(temp_dir, "audio_output")
        if not os.path.exists(audio_output_dir):
            os.makedirs(audio_output_dir)

        # Convert caption to audio
        text_to_audio(kokoro_model, caption, output_dir=audio_output_dir)

        # Read audio file
        wav_output_path = os.path.join(audio_output_dir, "audio_output.wav")
        if platform.system() == "Darwin":
            os.system(f"afplay {wav_output_path}")
            
        with open(wav_output_path, "rb") as f:
            audio_bytes = f.read()
            encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        # Clean up temp file
        try:
            os.remove(temp_file_path)
        except:
            pass

        return JSONResponse(content={
            "caption": caption,
            "audio_base64": encoded_audio
        })

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run the server when executing this file directly
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Server started. Models will be loaded on demand.")
    except Exception as e:
        logger.error(f"Startup event failed: {e}", exc_info=True)


if __name__ == "__main__":
    # Get port from environment variable or default to 8080
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("cloud_server:app", host="0.0.0.0", port=port)