#!/usr/bin/env python3
"""
Vision Processing Microservice
Terrain hazard classification using ConvNeXt-Tiny model
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import logging
import time
from pathlib import Path

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Vision Processing Service",
    description="Terrain hazard classification for Mars rover operations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
device = None
transform = None
MODEL_VERSION = "convnext-tiny-v1.0"

# Pydantic models
class ClassificationRequest(BaseModel):
    image_path: str
    return_confidence: bool = True

class ClassificationResponse(BaseModel):
    classification: str
    confidence: float
    model_version: str
    processing_time_ms: int

class BatchClassificationRequest(BaseModel):
    image_paths: List[str]

class ModelInfo(BaseModel):
    model_name: str
    version: str
    architecture: str
    parameters: int
    classes: List[str]
    loaded: bool


def load_model():
    """Load the ConvNeXt vision model"""
    global model, device, transform
    
    model_path = os.getenv('MODEL_PATH', '/models/terrain_vision_convnext.pth')
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=device)
            model.eval()
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
            model = None
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Vision Processing Service...")
    load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "service": "vision-service",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }


@app.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    return ModelInfo(
        model_name="Terrain Hazard Classifier",
        version=MODEL_VERSION,
        architecture="ConvNeXt-Tiny",
        parameters=28_000_000,  # Approximate
        classes=["SAFE", "CAUTION", "HAZARD"],
        loaded=model is not None
    )


@app.post("/classify_hazard", response_model=ClassificationResponse)
async def classify_hazard(file: UploadFile = File(...)):
    """Classify a single image for terrain hazards"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform and prepare for inference
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Map prediction to class
        classes = ["SAFE", "CAUTION", "HAZARD"]
        classification = classes[predicted.item()]
        confidence_value = confidence.item()
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Classified as {classification} with {confidence_value:.2%} confidence")
        
        return ClassificationResponse(
            classification=classification,
            confidence=confidence_value,
            model_version=MODEL_VERSION,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify_batch")
async def classify_batch(files: List[UploadFile] = File(...)):
    """Classify multiple images in batch"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            start_time = time.time()
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            classes = ["SAFE", "CAUTION", "HAZARD"]
            classification = classes[predicted.item()]
            
            results.append({
                "filename": file.filename,
                "classification": classification,
                "confidence": confidence.item(),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "model_version": MODEL_VERSION}


@app.post("/generate_hazard_map")
async def generate_hazard_map(request: Dict):
    """Generate a hazard map from DEM and vision analysis"""
    # Placeholder for future DEM integration
    return {
        "status": "not_implemented",
        "message": "DEM-based hazard map generation coming soon"
    }


@app.post("/reload_model")
async def reload_model():
    """Reload the model (for hot-swapping)"""
    try:
        load_model()
        return {
            "status": "success",
            "model_loaded": model is not None,
            "version": MODEL_VERSION
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
