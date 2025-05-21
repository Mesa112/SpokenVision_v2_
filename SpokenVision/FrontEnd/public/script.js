// DOM elements
const video = document.getElementById('video');
const startBtn = document.getElementById('startBtn');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const responseContainer = document.getElementById('responseContainer');
const audioToggle = document.getElementById('audioToggle');
const recordingStatus = document.getElementById('recordingStatus');
const modelLoadingStatus = document.getElementById('modelLoadingStatus');

// Global variables
let mediaRecorder;
let recordedChunks = [];
let videoStream;
let processingInterval;
let isProcessing = false;
let isRecording = false;
let facingMode = 'environment'; // 'environment' is back camera, 'user' is front camera
let modelsLoaded = false;
let apiVerified = false;
let speechEnabled = false;

// API endpoint
const API_ENDPOINT = "https://spokenvision-952306169360.us-central1.run.app"; //"https://spokenvision-952306169360.us-central1.run.app";

// Initialize model loader
let modelLoader;

// DOMContentLoaded

// TTS on first startBtn click (not DOMContentLoaded dependent)
// if (startBtn && !startBtn.hasSpeechListener) {
//     startBtn.addEventListener('click', () => {
//         if (!speechEnabled) {
//             speakText("Models loaded. You can start the camera.");
//             speechEnabled = true;
//         }
//     });
//     startBtn.hasSpeechListener = true;
// }

// Now wait for DOM to be ready for everything else

document.addEventListener('DOMContentLoaded', async () => {
    startBtn.disabled = true;
    recordBtn.disabled = true;
    stopBtn.disabled = true;

    startBtn.addEventListener('click', () => {
        if (videoStream) {
            stopSession();
        } else {
            startCamera();
        }
    });
    recordBtn.addEventListener('click', toggleRecording);
    stopBtn.addEventListener('click', stopSession);
    switchCameraBtn.addEventListener('click', switchCamera);

    modelLoader = new ModelLoader();

    addResponse('Loading AI models, please wait...', false, true);

    try {
        const success = await modelLoader.initializeModels();
        if (success) {
            modelsLoaded = true;
            //addResponse('AI models loaded successfully. Verifying API connection...', false, true);
            await verifyApiConnection();
            if (apiVerified) {
                addResponse('System ready! You can now start the camera.', false, true);
                startBtn.disabled = false;
            }
        }
    } catch (error) {
        console.error('Error initializing models:', error);
        addResponse(`Error loading models: ${error.message}. Please refresh the page to try again.`, false, true);
    }
});

// Verify API connection with a simple test
async function verifyApiConnection() {
    try {
        // Create a realistic test image (16x16 RGB gradients)
        const canvas = document.createElement('canvas');
        canvas.width = 16;
        canvas.height = 16;
        const ctx = canvas.getContext('2d');
        
        // Create a gradient that mimics a real image
        const gradient = ctx.createLinearGradient(0, 0, 16, 16);
        gradient.addColorStop(0, 'blue');
        gradient.addColorStop(0.5, 'green');
        gradient.addColorStop(1, 'red');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 16, 16);
        
        // Add some shapes to make it more like a real scene
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.fillRect(4, 4, 8, 8);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.beginPath();
        ctx.arc(8, 8, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        // Convert to blob
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
        
        // Create form data
        const formData = new FormData();
        formData.append("image", blob, "test.jpg");
        
        // Add verification message
        addResponse("Verifying API connection...", false, true);
        
        // Send test request with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000); // 15-second timeout
        
        try {
            const response = await fetch(`${API_ENDPOINT}/process/`, {
                method: "POST",
                body: formData,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }
            
            // Process result (just to verify format)
            const result = await response.json();
            
            if (result && (result.caption || result.audio_base64)) {
                apiVerified = true;
                addResponse("API connection successful! The system is ready to use.", false, true);
                return true;
            } else {
                throw new Error("Invalid response format from API");
            }
        } catch (fetchError) {
            // If there was an error with the API but it might be temporary
            // Still allow the user to try using the camera
            console.warn("API verification warning:", fetchError);
            addResponse(`Warning: API test gave an unexpected response: ${fetchError.message}. You can still try using the camera.`, false, true);
            
            // Consider the API verification "passed" with a warning
            apiVerified = true;
            return true;
        }
    } catch (err) {
        console.error("❌ API verification setup failed:", err);
        addResponse(`Error preparing API test: ${err.message}. You can try refreshing the page or continue anyway.`, false, true);
        
        // Let the user try anyway
        apiVerified = true;
        return true;
    }
}

// Start camera stream
async function startCamera() {
    if (!modelsLoaded || !apiVerified) {
        addResponse("Please wait for AI models to load completely and API to be verified.");
        return;
    }
    
    try {
        const constraints = {
            video: {
                facingMode: facingMode
            },
            audio: false
        };
        
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log("✅ Camera stream started");
        
        // Connect stream to video element
        video.srcObject = videoStream;
        
        // Update UI
        startBtn.innerHTML = '<i class="fas fa-stop me-2"></i>Stop Camera';
        recordBtn.disabled = false;
        switchCameraBtn.style.display = 'block';
        
        // Add initial message
        addResponse("Camera started. Point it at objects to get descriptions.");
        
        // Start processing frames every 5 seconds
        processingInterval = setInterval(captureAndSend, 5000);
    } catch (err) {
        console.error("❌ Camera access denied:", err);
        addResponse("Error: Could not access camera. Please check permissions.");
    }
}

// Switch between front and back cameras
async function switchCamera() {
    if (!videoStream) return;
    
    // Toggle facing mode
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    
    // Stop current stream
    videoStream.getTracks().forEach(track => track.stop());
    
    // Get new stream with different camera
    try {
        const constraints = {
            video: {
                facingMode: facingMode
            },
            audio: false
        };
        
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = videoStream;
        
        addResponse(`Switched to ${facingMode === 'environment' ? 'back' : 'front'} camera`);
    } catch (err) {
        console.error("❌ Error switching camera:", err);
        addResponse(`Error switching camera: ${err.message}`);
    }
}

// Toggle recording state
async function toggleRecording() {
    if (!isRecording) {
        // Start recording
        try {
            // Get audio stream for recording
            const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Combine video and audio streams
            const videoTrack = videoStream.getVideoTracks()[0];
            const audioTrack = audioStream.getAudioTracks()[0];
            const combinedStream = new MediaStream([videoTrack, audioTrack]);
            
            // Create media recorder
            mediaRecorder = new MediaRecorder(combinedStream, { mimeType: 'video/webm' });
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = () => {
                saveRecording();
            };
            
            // Start recording
            mediaRecorder.start();
            isRecording = true;
            
            // Update UI
            recordBtn.innerHTML = '<i class="fas fa-pause me-2"></i>Pause';
            recordBtn.classList.replace('btn-danger', 'btn-warning');
            stopBtn.disabled = false;
            recordingStatus.classList.remove('d-none');
            
            addResponse("Recording started. SpokenVision will continue to analyze the scene.");
            
        } catch (err) {
            console.error("❌ Error starting recording:", err);
            addResponse("Error: Could not start recording. Microphone access may be blocked.");
        }
    } else {
        // Pause recording
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.pause();
            recordBtn.innerHTML = '<i class="fas fa-record-vinyl me-2"></i>Resume';
            recordBtn.classList.replace('btn-warning', 'btn-success');
            recordingStatus.classList.add('d-none');
            addResponse("Recording paused.");
        } else if (mediaRecorder && mediaRecorder.state === 'paused') {
            mediaRecorder.resume();
            recordBtn.innerHTML = '<i class="fas fa-pause me-2"></i>Pause';
            recordBtn.classList.replace('btn-success', 'btn-warning');
            recordingStatus.classList.remove('d-none');
            addResponse("Recording resumed.");
        }
        isRecording = !isRecording;
    }
}

// Save the recorded video
function saveRecording() {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    
    // Create download link
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'spokenvision-demo-' + new Date().toISOString().slice(0,19).replace(/:/g,'-') + '.webm';
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }, 100);
    
    recordedChunks = [];
    addResponse("Recording saved. Check your downloads folder.");
}

// Stop all streams and recording
function stopSession() {
    // Stop recording if active
    if (mediaRecorder && (mediaRecorder.state === 'recording' || mediaRecorder.state === 'paused')) {
        mediaRecorder.stop();
        isRecording = false;
    }

    // Stop all streams
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }

    // Clear interval
    clearInterval(processingInterval);

    // Reset UI
    video.srcObject = null;
    startBtn.disabled = false;
    startBtn.innerHTML = '<i class="fas fa-play me-2"></i>Start Camera';
    recordBtn.disabled = true;
    stopBtn.disabled = true;
    switchCameraBtn.style.display = 'none';

    // Reset record button
    recordBtn.innerHTML = '<i class="fas fa-record-vinyl me-2"></i>Record Demo';
    recordBtn.classList.replace('btn-warning', 'btn-danger');
    recordBtn.classList.replace('btn-success', 'btn-danger');
    recordingStatus.classList.add('d-none');

    // Reset flags
    isRecording = false;
    recordedChunks = [];

    addResponse("Session ended. If you recorded a demo, it should download automatically.");
}


// Capture video frame and send to server
async function captureAndSend() {
    // Prevent multiple simultaneous processing requests
    if (isProcessing || !videoStream || !videoStream.active) return;
    
    isProcessing = true;
    console.log("📸 Capturing frame...");
    
    try {
        // Create canvas to capture frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        // Convert to blob
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
        
        // Create form data
        const formData = new FormData();
        formData.append("image", blob, "frame.jpg");
        
        // Show loading message
        const loadingId = addResponse("Processing image...", true);
        
        // Send to server
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            body: formData
        });
        
        // Remove loading message
        removeResponse(loadingId);
        
        if (!response.ok) {
            const errText = await response.text();
            console.error("❌ Server responded with error:", response.status, errText);
            addResponse(`Error: Server responded with status ${response.status}`);
            isProcessing = false;
            return;
        }
        
        // Process result
        const result = await response.json();
        console.log("✅ Server responded:", result.caption);
        
        // Add response to UI
        addResponse(result.caption);
        
        // Play audio if enabled
        if (audioToggle.checked && result.audio_base64) {
            playAudio(result.audio_base64);
        }
    } catch (err) {
        console.error("❌ Request failed:", err);
        addResponse("Error: Failed to process image. Server might be unavailable.");
    } finally {
        isProcessing = false;
    }
}

// Function to play audio from base64 data
function playAudio(base64Data) {
    const audio = new Audio(`data:audio/mp3;base64,${base64Data}`);
    audio.play().catch(err => {
        console.error('Error playing audio:', err);
        addResponse("Audio playback failed. You may need to interact with the page first.");
    });
}

// Add response message to UI
function addResponse(text, isLoading = false, replaceExisting = false) {
    // Create response item
    const item = document.createElement('div');
    item.className = 'response-item';
    
    const id = 'response-' + Date.now();
    item.id = id;
    
    const timestamp = new Date().toLocaleTimeString();
    
    if (isLoading) {
        item.innerHTML = `
            <div class="d-flex align-items-center">
                <div class="spinner-border spinner-border-sm text-primary me-2" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mb-0">${text}</p>
            </div>
            <div class="timestamp">${timestamp}</div>
        `;
    } else {
        item.innerHTML = `
            <p>${text}</p>
            <div class="timestamp">${timestamp}</div>
        `;
    }
    
    // Clear container if replace is true
    if (replaceExisting) {
        responseContainer.innerHTML = '';
    }
    
    // Add to container
    responseContainer.appendChild(item);
    responseContainer.scrollTop = responseContainer.scrollHeight; // Auto-scroll to bottom
    
    // Only use browser TTS for status messages during initialization
    // Once models are loaded and camera is active, we'll rely on Kokoro TTS for actual scene descriptions
    if (audioToggle && audioToggle.checked && !isLoading) {
        // Determine if this is a system message that should use browser TTS
        const isSystemMessage = 
            text.startsWith("Loading") || 
            text.startsWith("AI models") || 
            text.startsWith("Verifying") || 
            text.startsWith("System ready") || 
            text.startsWith("Error") || 
            text.startsWith("Warning") || 
            text.startsWith("Camera") || 
            text.startsWith("Recording") || 
            text.startsWith("Session ended") || 
            text.startsWith("Switched to");
        
        // Only use browser TTS for system messages
        // For actual scene descriptions (after camera is active), we rely on Kokoro TTS from the backend
        if (isSystemMessage) {
            speakText(text);
        }
    }
    
    return id;
}

// Function to speak text using browser text-to-speech
function speakText(text) {
    // Cancel any previous speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    window.speechSynthesis.speak(utterance);
}

// Remove a response element by ID
function removeResponse(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}
