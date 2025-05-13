FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*

WORKDIR /app

COPY docker_requirements.txt .
RUN pip install --no-cache-dir -r docker_requirements.txt \
    && pip install --no-cache-dir \
        transformers \
        huggingface_hub \
        espnet \
        ipython \
        pydantic \
        gtts

# Preload segmentation model
RUN python -c "\
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor; \
SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512', cache_dir='/root/.cache/huggingface'); \
SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512', cache_dir='/root/.cache/huggingface')"

# Preload depth model
RUN python -c "\
from transformers import DPTImageProcessor, DPTForDepthEstimation; \
DPTImageProcessor.from_pretrained('Intel/dpt-hybrid-midas', cache_dir='/root/.cache/huggingface'); \
DPTForDepthEstimation.from_pretrained('Intel/dpt-hybrid-midas', cache_dir='/root/.cache/huggingface')"

# Preload BLIP captioning model
RUN python -c "\
from transformers import BlipProcessor, BlipForConditionalGeneration; \
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large', cache_dir='/root/.cache/huggingface'); \
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large', cache_dir='/root/.cache/huggingface')"

# Preload Kokoro voice model with correct language code
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='hexgrad/Kokoro-82M', filename='config.json', cache_dir='/root/.cache/huggingface')"

# Use Facebook's MMS TTS model which is more reliable than the espnet one
RUN python -c "\
from transformers import AutoProcessor, MmsTextToWaveformModel; \
processor = AutoProcessor.from_pretrained('facebook/mms-tts-eng', cache_dir='/root/.cache/huggingface'); \
model = MmsTextToWaveformModel.from_pretrained('facebook/mms-tts-eng', cache_dir='/root/.cache/huggingface')"

COPY . .

EXPOSE 8080
CMD ["uvicorn", "cloud_server:app", "--host", "0.0.0.0", "--port", "8080"]