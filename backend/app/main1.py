from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers.models.wav2vec2 import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import os
import tempfile
from .audio_detection import predict_audio
from .config import logger, MODEL_PATH
from starlette.requests import Request
from pathlib import Path
from contextlib import asynccontextmanager

# Global variables for model and processor
model = None
processor = None

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events.
    Load the model and processor at startup.
    """
    global model, processor
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
        processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown logic
    logger.info("Shutting down application")

app = FastAPI(title="Audio Deepfake Detection", lifespan=lifespan)

# Mount templates directory
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

@app.get("/", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    """
    Serve the HTML upload page.
    """
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/audio/detect")
async def detect_audio(file: UploadFile = File(...)):
    """
    Detect if an audio file is real or fake.
    Returns prediction, confidence, and path to saved processed WAV file.
    """
    try:
        # Validate file extension
        if not file.filename or not file.filename.lower().endswith(('.wav', '.mp3')):
            raise HTTPException(status_code=400, detail="Unsupported file format. Only .wav and .mp3 are supported.")
        
        # Save uploaded file to temporary location
        filename = file.filename or "default.wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # Predict
        label, confidence, saved_wav_path = predict_audio(temp_file_path, model, processor)
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {temp_file_path}: {str(e)}")
        
        return {
            "prediction": label,
            "confidence": confidence,
            "saved_wav_path": saved_wav_path
        }
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")