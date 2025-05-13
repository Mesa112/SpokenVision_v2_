#!/usr/bin/env python3
"""
SpokenVision Continuous Webcam Test

This script runs continuously with the webcam and:
1. Captures an image every 15 frames
2. Processes it through the entire SpokenVision pipeline
3. Speaks the resulting description
4. Repeats until the user presses ESC

Usage:
    python continuous_webcam_test.py [--silent] [--interval FRAMES]
"""

import argparse
import cv2
import numpy as np
import os
import tempfile
import time
import threading
from PIL import Image
import soundfile as sf

# Parse command line arguments
parser = argparse.ArgumentParser(description='SpokenVision Continuous Webcam Test')
parser.add_argument('--silent', action='store_true', help='Do not play audio')
parser.add_argument('--interval', type=int, default=15, help='Process every N frames (default: 15)')
args = parser.parse_args()

# Global variables
processing_in_progress = False
audio_playing = False
latest_frame = None
latest_description = "No description available yet."

def process_image(frame_rgb):
    """Process an image through the entire SpokenVision pipeline"""
    global processing_in_progress, audio_playing, latest_description
    
    processing_in_progress = True
    print("\n===== Processing new frame =====")
    
    try:
        # Step 1: Object Detection
        from object_detection import load_model, detect_objects
        detection_model = load_model()
        detection_results = detect_objects(frame_rgb, detection_model, conf_threshold=0.3)
        print(f"✅ Object detection: Found {len(detection_results['boxes'])} objects")
        
        # Step 2: Depth Estimation
        from depth_estimation import load_depth_model, estimate_depth
        depth_model = load_depth_model()
        depth_map = estimate_depth(frame_rgb, depth_model)
        print("✅ Depth estimation completed")
        
        # Step 3: Load BLIP and generate caption
        from blip_image_captioning import load_blip_captioning_model, generate_caption
        blip_model = load_blip_captioning_model()
        caption = generate_caption(frame_rgb, blip_model)
        print(f"✅ Caption: \"{caption}\"")
        
        # Step 4: Try to load segmentation (optional)
        try:
            from semantic_segmentation import load_model as load_segmentation_model
            from semantic_segmentation import predict_segmentation
            import torch.nn.functional as F
            
            # Save frame to temp file for segmentation
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file_path = temp_file.name
                cv2.imwrite(temp_file_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                
            seg_model, feat_extractor = load_segmentation_model()
            segmentation_map = predict_segmentation(temp_file_path, seg_model, feat_extractor)
            
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except:
                pass
                
            print("✅ Segmentation completed")
        except Exception as e:
            print(f"⚠️ Segmentation failed (non-critical): {e}")
            segmentation_map = None
        
        # Step 5: Context Building
        from context_builder import ContextBuilder
        context_builder = ContextBuilder()
        
        # Handle segmentation map
        if segmentation_map is not None:
            frame_height, frame_width = frame_rgb.shape[:2]
            # Convert to frame dimensions
            import torch.nn.functional as F
            latest_segmentation_map = F.interpolate(
                segmentation_map.unsqueeze(0).unsqueeze(0).float(),
                size=(frame_height, frame_width),
                mode='nearest'
            ).squeeze().cpu().numpy()
        else:
            # Create dummy segmentation map
            latest_segmentation_map = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
        
        # Build context description
        context_description = context_builder.process_frame_data(
            detection_results, 
            depth_map, 
            latest_segmentation_map, 
            caption
        )
        print("✅ Context description generated")
        print("\n===== CONTEXT DESCRIPTION =====")
        print(context_description)
        print("===============================\n")
        
        # Step 6: Enhance with GPT (simulate this step - unimplemented in local mode)
        # For real implementation, you would call the OpenAI API here
        from cloud_server import enhance_description_with_gpt
        try:
            enhanced_description = enhance_description_with_gpt(context_description)
            print("✅ GPT enhancement completed")
            if enhanced_description:
                context_description = enhanced_description
                print("Enhanced description: ", enhanced_description)
        except Exception as e:
            print(f"⚠️ GPT enhancement failed (non-critical): {e}")
        
        # Save the latest description
        latest_description = context_description
        
        # Step 7: Text-to-Speech
        if not args.silent and not audio_playing:
            # Create a thread for audio so it doesn't block processing
            audio_thread = threading.Thread(
                target=play_audio_description,
                args=(context_description,)
            )
            audio_thread.daemon = True
            audio_thread.start()
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False
    finally:
        processing_in_progress = False

def play_audio_description(description_text):
    """Generate and play TTS audio for the description"""
    global audio_playing
    
    audio_playing = True
    try:
        from kokoro_audio import load_kokoro_model, text_to_audio
        
        # Load TTS model
        kokoro_model = load_kokoro_model()
        
        # Generate audio
        temp_dir = tempfile.mkdtemp()
        audio_files = text_to_audio(kokoro_model, description_text, output_dir=temp_dir)
        
        # Play the audio
        if audio_files:
            import platform
            system = platform.system()
            
            for audio_file in audio_files:
                if system == "Darwin":  # macOS
                    os.system(f"afplay {audio_file}")
                elif system == "Linux":
                    os.system(f"aplay {audio_file}")
                elif system == "Windows":
                    import winsound
                    winsound.PlaySound(audio_file, winsound.SND_FILENAME)
                else:
                    print(f"⚠️ Unsupported OS for audio playback: {system}")
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
    finally:
        audio_playing = False

def main():
    print("===== SpokenVision Continuous Webcam Test =====")
    print(f"Processing every {args.interval} frames")
    print("Press ESC to quit")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    # Variables for frame counting
    frame_count = 0
    last_processed_time = time.time()
    
    # Main loop
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from webcam")
            break
        
        # Store latest frame globally
        global latest_frame
        latest_frame = frame.copy()
        
        # Increment frame counter
        frame_count += 1
        
        # Display status on frame
        status_text = f"Frame: {frame_count}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if processing_in_progress:
            status = "Processing..."
            color = (0, 0, 255)  # Red
        elif audio_playing:
            status = "Speaking..."
            color = (255, 165, 0)  # Orange
        else:
            status = "Ready"
            color = (0, 255, 0)  # Green
            
        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display the latest description (truncated to fit on screen)
        desc_lines = latest_description.split("\n")
        short_desc = desc_lines[0][:50] + "..." if len(desc_lines[0]) > 50 else desc_lines[0]
        cv2.putText(frame, short_desc, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('SpokenVision Continuous Test (ESC to quit)', frame)
        
        # Process a frame every N frames if not already processing
        if frame_count % args.interval == 0 and not processing_in_progress and not audio_playing:
            # Calculate time since last processing
            current_time = time.time()
            time_since_last = current_time - last_processed_time
            
            # Only process if enough time has passed (at least 2 seconds)
            if time_since_last >= 2:
                # Convert BGR to RGB (OpenCV uses BGR, our models expect RGB)
                frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
                
                # Process in a separate thread to not block the UI
                process_thread = threading.Thread(
                    target=process_image,
                    args=(frame_rgb,)
                )
                process_thread.daemon = True
                process_thread.start()
                
                # Update last processed time
                last_processed_time = current_time
        
        # Check for exit key (ESC)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            print("👋 Exiting...")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Exiting.")

if __name__ == "__main__":
    main()