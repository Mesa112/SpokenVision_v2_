# SpokenVision

**SpokenVision** is an assistive AI application designed to help visually impaired users perceive their environment through real-time audio descriptions. It captures video input, detects and identifies objects, estimates their distance, and converts the information into spoken feedback using advanced AI models.

---

## 🔍 Key Features

- Real-time object detection (YOLOv8)
- Depth estimation using MiDaS or DPT
- Scene captioning using BLIP or Qwen
- Natural language understanding (ChatGPT integration)
- Audio synthesis using Kokoro TTS
- Frontend using Streamlit

---

## 📸 Project Flow

1. **Camera Feed** captures real-time video.
2. **`object_detection.py`** identifies objects using YOLOv8.
3. **`depth_estimation.py`** computes depth using MiDaS.
4. **`blip_image_captioning.py` / `qwen_captioning.py`** generate captions.
5. **`context_builder.py` / `position_estimator.py`** merge object info with spatial context.
6. **`model_inference.py` / `main.py`** integrate models and manage control flow.
7. **`kokoro_audio.py`** and **`speech_Output.py`** convert text output to speech.
8. **`streamlit.py` / `server.py` / `object_detection.py`** serve the app frontend.

---

## 📁 File Overview

| File / Folder              | Purpose                                                                  |
|----------------------------|-------------------------------------------------------------------------|
| `main.py`                  | Main entry point for backend orchestration                              |
| `object_detection.py`      | YOLOv8-based object detection                                           |
| `depth_estimation.py`      | Runs MiDaS depth model                                                  |
| `position_estimator.py`    | Merges object detection with estimated distance                         |
| `blip_image_captioning.py` | BLIP model-based image captioning                                       |
| `qwen_captioning.py`       | Qwen captioning model integration                                       |
| `model_inference.py`       | Handles AI model execution pipelines                                    |
| `kokoro_audio.py`          | Text-to-speech engine for spoken feedback (Kokoro)                      |
| `speech_Output.py`         | Generates and manages audio output                                      |
| `context_builder.py`       | Combines detection, captioning, and depth info                          |
| `server.py`                | Backend server for communication with frontend                          |
| `streamlit.py`             | Streamlit UI for user interaction                                       |
| `test.ipynb`               | Notebook for testing server + models                                    |
| `semantic_segmentation.py` | (Optional) Semantic segmentation pipeline                               |
| `requirements.txt`         | Python dependencies                                                     |
| `yolov8n.pt`               | YOLOv8 weights                                                          |

----

## ⚙️ Setup Instructions

### 🔧 Prerequisites

- Python 3.8+
- Node.js (optional if using additional UI)
- pip / conda
- FFmpeg (for audio)
- GPU recommended (for YOLO and MiDaS)

### 🧪 Installation

```bash
git clone https://github.com/Mesa112/SpokenVision.git
cd SpokenVision

# Python environment
pip install -r requirements.txt
