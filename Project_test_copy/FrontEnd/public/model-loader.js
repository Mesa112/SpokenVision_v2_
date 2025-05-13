// model-loader.js
// This file demonstrates how to handle model loading for SpokenVision

class ModelLoader {
    constructor() {
        this.models = {
            objectDetection: { name: 'Object Detection (YOLO)', loaded: false, status: 'pending' },
            depthEstimation: { name: 'Depth Estimation', loaded: false, status: 'pending' },
            textRecognition: { name: 'Text Recognition (OCR)', loaded: false, status: 'pending' },
            nlpProcessor: { name: 'Natural Language Processing', loaded: false, status: 'pending' },
            gptEnhancement: { name: 'GPT Text Enhancement', loaded: false, status: 'pending' } // Add GPT model
        };
        
        this.modelStatusElement = document.getElementById('modelLoadingStatus');
        this.startBtn = document.getElementById('startBtn');
    }
    
    // Initialize all models
    async initializeModels() {
        this.updateModelStatusUI();
        
        try {
            // Load models in parallel
            await Promise.all([
                this.loadObjectDetectionModel(),
                this.loadDepthEstimationModel(),
                this.loadTextRecognitionModel(),
                this.loadNLPModel(),
                this.testGPTEnhancement() // Add GPT test
            ]);
            
            // If all models are loaded, enable the start button
            if (this.allModelsLoaded()) {
                this.modelStatusElement.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle status-icon status-success"></i>
                        All models loaded successfully. You can now start the camera.
                    </div>
                `;
                this.startBtn.disabled = false;
                return true;
            }
        } catch (error) {
            console.error('Error initializing models:', error);
            this.modelStatusElement.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle status-icon status-error"></i>
                    Error loading models. Please refresh the page and try again.
                </div>
            `;
            return false;
        }
    }
    
    // Check if all models are loaded
    allModelsLoaded() {
        return Object.values(this.models).every(model => model.loaded);
    }
    
    // Update the UI to show model loading status
    updateModelStatusUI() {
        let statusHTML = '<div class="mb-3">Loading required AI models:</div>';
        
        // Create status items for each model
        Object.values(this.models).forEach(model => {
            let statusIcon, statusClass;
            
            switch (model.status) {
                case 'success':
                    statusIcon = 'check-circle';
                    statusClass = 'status-success';
                    break;
                case 'error':
                    statusIcon = 'exclamation-triangle';
                    statusClass = 'status-error';
                    break;
                default:
                    statusIcon = 'spinner fa-spin';
                    statusClass = 'status-pending';
            }
            
            statusHTML += `
                <div class="d-flex align-items-center mb-2">
                    <i class="fas fa-${statusIcon} status-icon ${statusClass} me-2"></i>
                    <span>${model.name}</span>
                    <span class="ms-auto">${model.loaded ? 'Loaded' : 'Loading...'}</span>
                </div>
            `;
        });
        
        this.modelStatusElement.innerHTML = statusHTML;
    }
    
    // Test GPT Enhancement API
    // Test GPT Enhancement API
async testGPTEnhancement() {
    try {
        // Update status
        this.models.gptEnhancement.status = 'pending';
        this.updateModelStatusUI();
        
        // Create AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
        
        // Simple test request to API endpoint with test prompt
        const response = await fetch("https://spokenvision-952306169360.us-central1.run.app/test-gpt/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: "This is a test prompt to verify GPT integration."
            }),
            signal: controller.signal
        });
        
        // Clear the timeout
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`GPT test failed with status ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result && result.success) {
            this.models.gptEnhancement.loaded = true;
            this.models.gptEnhancement.status = 'success';
            console.log("GPT enhancement test successful");
        } else {
            // Handle partial success case (API responded but indicated failure)
            const errorMessage = result.error || "Unknown error";
            console.warn("GPT test responded but indicated failure:", errorMessage);
            this.models.gptEnhancement.status = 'error';
            
            // Add a user-facing message about GPT
            this.showGptWarningMessage(errorMessage);
            
            // Make GPT non-blocking by marking it as loaded even if test fails
            this.models.gptEnhancement.loaded = true;
        }
    } catch (error) {
        console.error('Error testing GPT enhancement:', error);
        this.models.gptEnhancement.status = 'error';
        
        // Add a user-facing message about GPT
        this.showGptWarningMessage(error.message);
        
        // Make GPT non-blocking by marking it as loaded even if test fails
        // This allows the app to function without GPT enhancement
        this.models.gptEnhancement.loaded = true;
    } finally {
        this.updateModelStatusUI();
    }
}

// Helper method to show a warning message about GPT
showGptWarningMessage(errorDetails) {
    // Create a warning message element
    const warningDiv = document.createElement('div');
    warningDiv.className = 'alert alert-warning mt-3';
    warningDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        <strong>GPT Enhancement Unavailable:</strong> 
        The app will continue to function, but text descriptions may be more technical.
        <div class="small text-muted mt-1">Error: ${errorDetails}</div>
    `;
    
    // Add to the model loading status container
    this.modelStatusElement.appendChild(warningDiv);
    
    // Also add to the response container for visibility
    if (window.addResponse) {
        window.addResponse("Note: GPT text enhancement is unavailable. Descriptions will be more technical but the app will function normally.", false, false);
    }
}
    
    // Load Object Detection Model (YOLO)
    async loadObjectDetectionModel() {
        try {
            // Simulate loading with a delay
            // In reality, you would use something like:
            // const model = await tf.loadGraphModel('path/to/your/model.json');
            await this.simulateModelLoading(2000);
            
            this.models.objectDetection.loaded = true;
            this.models.objectDetection.status = 'success';
        } catch (error) {
            console.error('Error loading object detection model:', error);
            this.models.objectDetection.status = 'error';
            throw error;
        } finally {
            this.updateModelStatusUI();
        }
    }
    
    // Load Depth Estimation Model
    async loadDepthEstimationModel() {
        try {
            await this.simulateModelLoading(3000);
            
            this.models.depthEstimation.loaded = true;
            this.models.depthEstimation.status = 'success';
        } catch (error) {
            console.error('Error loading depth estimation model:', error);
            this.models.depthEstimation.status = 'error';
            throw error;
        } finally {
            this.updateModelStatusUI();
        }
    }
    
    // Load Text Recognition Model (OCR)
    async loadTextRecognitionModel() {
        try {
            await this.simulateModelLoading(2500);
            
            this.models.textRecognition.loaded = true;
            this.models.textRecognition.status = 'success';
        } catch (error) {
            console.error('Error loading text recognition model:', error);
            this.models.textRecognition.status = 'error';
            throw error;
        } finally {
            this.updateModelStatusUI();
        }
    }
    
    // Load NLP Model
    async loadNLPModel() {
        try {
            await this.simulateModelLoading(1800);
            
            this.models.nlpProcessor.loaded = true;
            this.models.nlpProcessor.status = 'success';
        } catch (error) {
            console.error('Error loading NLP model:', error);
            this.models.nlpProcessor.status = 'error';
            throw error;
        } finally {
            this.updateModelStatusUI();
        }
    }
    
    // Simulate model loading with a promise and delay
    simulateModelLoading(delay) {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve();
            }, delay);
        });
    }
}