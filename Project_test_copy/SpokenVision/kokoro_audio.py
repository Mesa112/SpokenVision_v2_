from kokoro import KPipeline
import soundfile as sf
import os
import tempfile
import pathlib
import time
import numpy as np


def load_kokoro_model(lang_code='a'):
    """Load the Kokoro TTS model with the specified language code."""
    try:
        # Basic initialization exactly as shown in the documentation
        pipeline = KPipeline(lang_code=lang_code)
        print(f"✅ Kokoro model initialized with language code: {lang_code}")
        return {"type": "kokoro", "model": pipeline}
    except Exception as e:
        print(f"⚠️ Failed to load Kokoro model: {e}")
        print("✅ Trying Hugging Face TTS fallback...")
        
        # Try to load a Hugging Face TTS model as first fallback
        try:
            from transformers import AutoProcessor, AutoModelForTextToWaveform
            
            # Try a few different models in case one fails
            hf_models = [
                # Facebook's MMS TTS model - good quality, medium size
                ("facebook/mms-tts-eng", "mms"),
                # Microsoft's SpeechT5 model - good quality, larger
                ("microsoft/speecht5_tts", "speecht5"),
                # VITS model - smaller, faster
                ("espnet/kan-bayashi_ljspeech_vits", "vits")
            ]
            
            for model_name, model_type in hf_models:
                try:
                    print(f"Loading {model_name}...")
                    if model_type == "mms":
                        processor = AutoProcessor.from_pretrained(model_name)
                        model = AutoModelForTextToWaveform.from_pretrained(model_name)
                        print(f"✅ Successfully loaded {model_name}")
                        return {"type": "hf", "model": model, "processor": processor, "model_type": model_type}
                    
                    elif model_type == "speecht5":
                        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
                        processor = SpeechT5Processor.from_pretrained(model_name)
                        model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
                        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
                        print(f"✅ Successfully loaded {model_name}")
                        return {"type": "hf", "model": model, "processor": processor, 
                                "vocoder": vocoder, "model_type": model_type}
                    
                    elif model_type == "vits":
                        from transformers import AutoTokenizer, VitsModel
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = VitsModel.from_pretrained(model_name)
                        print(f"✅ Successfully loaded {model_name}")
                        return {"type": "hf", "model": model, "processor": tokenizer, "model_type": model_type}
                    
                except Exception as model_err:
                    print(f"⚠️ Failed to load {model_name}: {model_err}")
                    continue
            
            print("⚠️ All Hugging Face models failed to load")
            
        except ImportError as imp_err:
            print(f"⚠️ Cannot import Hugging Face models: {imp_err}")
        
        print("✅ Falling back to gTTS.")
        return {"type": "gtts", "model": None}

def text_to_audio(model_info, input_text, output_dir='./audio_output', file_name="audio_output", voice='af_heart'):
    """Convert text to audio using the specified model and voice."""
    # Make sure the output directory exists
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    except Exception:
        # Use a temp directory if there are problems
        output_dir = tempfile.mkdtemp()
        print(f"Using temporary directory for audio output: {output_dir}")
    
    audio_file = f"{output_dir}/{file_name}.wav"
    
    # Check which model type we have
    model_type = model_info.get("type", "gtts")
    
    # 1. Kokoro TTS (primary option)
    if model_type == "kokoro":
        model = model_info["model"]
        try:
            # List of voices to try in order
            voice_options = [
                voice,  # Try the requested voice first
                'af_heart', 'af_alloy', 'af_nova',  # English female voices
                'am_adam', 'am_echo', 'am_onyx'     # English male voices
            ]
            
            # Try each voice until one works
            for voice_name in voice_options:
                try:
                    print(f"Trying Kokoro with voice: {voice_name}")
                    generator = model(input_text, voice=voice_name)
                    all_audio = []
                    # Process all segments
                    for i, (gs, ps, audio) in enumerate(generator):
                        print(f"Segment {i}, audio length: {len(audio)}")
                        all_audio.append(audio) 
                        
                    combined_audio = np.concatenate(all_audio, axis=0)
                    sf.write(audio_file, combined_audio, 24000)
                    # if audio_file:
                    print(f"✅ Audio saved using Kokoro with voice {voice_name}")
                    return audio_file
                except Exception as e:
                    print(f"⚠️ Voice {voice_name} failed: {e}")
                    continue
            
            # If all voices failed, fall back to next option
            raise Exception("All Kokoro voices failed")
        
        except Exception as e:
            print(f"⚠️ Kokoro generation failed: {e}")
            print("Trying Hugging Face TTS fallback...")
            
            # Try to load a Hugging Face TTS model for fallback
            try:
                from transformers import AutoProcessor, AutoModelForTextToWaveform
                model_name = "facebook/mms-tts-eng"
                processor = AutoProcessor.from_pretrained(model_name)
                hf_model = AutoModelForTextToWaveform.from_pretrained(model_name)
                
                # Update model_info to use this fallback
                model_info = {
                    "type": "hf", 
                    "model": hf_model, 
                    "processor": processor, 
                    "model_type": "mms"
                }
                model_type = "hf"
                print(f"Loaded fallback model: {model_name}")
            except Exception as hf_err:
                print(f"⚠️ Couldn't load Hugging Face fallback: {hf_err}")
                # Fall through to gTTS
                model_type = "gtts"
    
    # 2. Hugging Face TTS (first fallback)
    if model_type == "hf":
        try:
            print(f"Using Hugging Face TTS model ({model_info['model_type']})...")
            
            if model_info["model_type"] == "mms":
                # MMS TTS model
                inputs = model_info["processor"](text=input_text, return_tensors="pt")
                
                import torch
                with torch.no_grad():
                    output = model_info["model"].generate_speech(**inputs)
                
                # Save the audio
                sf.write(audio_file, output.numpy(), 16000)
                print(f"✅ Audio saved to {audio_file} using Hugging Face MMS TTS")
                return [audio_file]
            
            elif model_info["model_type"] == "speecht5":
                # SpeechT5 model
                import torch
                
                # Generate input
                inputs = model_info["processor"](text=input_text, return_tensors="pt")
                
                # Default speaker embedding
                speaker_embeddings = torch.zeros((1, 512))
                
                # Generate speech
                speech = model_info["model"].generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings=speaker_embeddings,
                    vocoder=model_info["vocoder"]
                )
                
                # Save the audio
                sf.write(audio_file, speech.numpy(), 16000)
                print(f"✅ Audio saved to {audio_file} using Hugging Face SpeechT5")
                return [audio_file]
            
            elif model_info["model_type"] == "vits":
                # VITS model
                import torch
                
                # Generate input
                inputs = model_info["processor"](text=input_text, return_tensors="pt")
                
                # Generate speech
                with torch.no_grad():
                    output = model_info["model"](
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"]
                    )
                
                # Save the audio
                sf.write(audio_file, output.waveform[0].numpy(), model_info["model"].config.sampling_rate)
                print(f"✅ Audio saved to {audio_file} using Hugging Face VITS")
                return [audio_file]
            
            else:
                # Unknown model type, fall back to gTTS
                raise ValueError(f"Unknown model type: {model_info['model_type']}")
                
        except Exception as hf_err:
            print(f"⚠️ Hugging Face TTS generation failed: {hf_err}")
            print("Falling back to gTTS...")
            model_type = "gtts"  # Fall through to gTTS
    
    # 3. Google Text-to-Speech (final fallback)
    if model_type == "gtts":
        try:
            from gtts import gTTS
            tts = gTTS(input_text)
            tts.save(audio_file)
            print(f"✅ Audio saved to {audio_file} using gTTS")
            return [audio_file]
        except Exception as gtt_err:
            print(f"⚠️ Final gTTS fallback failed: {gtt_err}")
            # Create a silent audio file as last resort
            empty_audio = np.zeros(16000)  # 1 second of silence at 16kHz
            sf.write(audio_file, empty_audio, 16000)
            print(f"⚠️ Created silent audio file: {audio_file}")
            return [audio_file]