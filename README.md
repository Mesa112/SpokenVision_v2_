# [SpokenVision](https://spokenvision.web.app/)

Demo/Ad

<p align="center">
  <a href="https://youtube.com/shorts/gWyHpFkxIAg?feature=share">
    <img src="https://i.imgur.com/fDb4ilz.png" style="max-height: 500px; width: auto;" />
  </a>
</p>

[SpokenVision Slides Presentation](https://docs.google.com/presentation/d/1-q8HFq-ZHlbFR2NloiWQXmuxu3pfY8jJQWInpOJ0tsQ/edit?usp=sharing)
[Deployed App](https://spokenvision.web.app/)

SpokenVision is an assistive AI application designed to help visually impaired users perceive their environment through real-time audio descriptions. It captures video input, detects and identifies objects, estimates their distance, and converts the information into spoken feedback using advanced AI models.
## 🔍 Key Features

    Real-time object detection (YOLOv8)
    Depth estimation using MiDaS
    Scene captioning using BLIP or Qwen
    Natural language understanding (ChatGPT integration)
    Audio synthesis using gTTS (Google Text-to-Speech)
    Frontend using React.js

## 📸 Project Flow
    ![SpokenVision Workflow Diagram](https://i.imgur.com/ImyREFn.png)

    Camera Feed captures real-time video.
    object_detection.py identifies objects using YOLOv8.
    depth_estimation.py computes depth using MiDaS.
    blip_image_captioning.py / qwen_captioning.py generate captions.
    context_builder.py / position_estimator.py merge object info with spatial context.
    model_inference.py / main.py integrate models and manage control flow.
    kokoro_audio.py and speech_Output.py convert text output to speech.
    streamlit.py / server.py / object_detection.py serve the app frontend.

## 📁 File Overview
File / Folder 	Purpose
main.py 	Main entry point for backend orchestration
object_detection.py 	YOLOv8-based object detection
depth_estimation.py 	Runs MiDaS depth model
position_estimator.py 	Merges object detection with estimated distance
blip_image_captioning.py 	BLIP model-based image captioning
qwen_captioning.py 	Qwen captioning model integration
model_inference.py 	Handles AI model execution pipelines
kokoro_audio.py 	Text-to-speech engine for spoken feedback (Kokoro)
speech_Output.py 	Generates and manages audio output
context_builder.py 	Combines detection, captioning, and depth info
server.py 	Backend server for communication with frontend
streamlit.py 	Streamlit UI for user interaction
test.ipynb 	Notebook for testing server + models
semantic_segmentation.py 	(Optional) Semantic segmentation pipeline
requirements.txt 	Python dependencies
yolov8n.pt 	YOLOv8 weights
# 🔧 Prerequisites

    Python 3.8+
    Node.js (optional if using additional UI)
    pip / conda
    FFmpeg (for audio)
    GPU recommended (for YOLO and MiDaS)

## 🧪 Installation

git clone https://github.com/Mesa112/SpokenVision.git
cd SpokenVision

## Python environment
pip install -r requirements.txt

## ⚙️ Setup Instructions

### 🖥️ Frontend

Serve the frontend using Python's built-in HTTP server:

```bash
cd SpokenVision/FrontEnd/public
python -m http.server 8000
```

### 🧠 Backend

Serve the backend using Uvicorn:

```bash
cd Backend
uvicorn cloud_server:app --host 0.0.0.0 --port 5000
```

### 🔁 Connect Frontend to Backend

Open the following files:

- `FrontEnd/public/script.js`
- `FrontEnd/public/model-loader.js`

Set the `API_ENDPOINT` to:

```js
const API_ENDPOINT = "http://localhost:5000";
```

- `Webpage will be hosted at http://localhost:8000/index.html`

