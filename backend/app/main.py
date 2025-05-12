

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
import uuid
from pathlib import Path
from transformers.models.wav2vec2 import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from .audio_detection import predict_audio
from .image_detection import detect_image_deepfake
from .config import AUDIO_MODEL_PATH, TEST_DATA_DIR, logger

app = FastAPI()

# Mount templates
templates = Jinja2Templates(directory="templates")

# Ensure test_data directory exists
Path(TEST_DATA_DIR).mkdir(parents=True, exist_ok=True)

# Load audio model and processor globally
try:
    audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_PATH)
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL_PATH)
    audio_model.eval()
    logger.info(f"Loaded audio model and processor from {AUDIO_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load audio model or processor from {AUDIO_MODEL_PATH}: {e}")
    raise

@app.get("/", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    """
    Serve the upload HTML page.
    """
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/audio/detect")
async def audio_detect(file: UploadFile = File(...)):
    """
    Endpoint for audio deepfake detection.
    """
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Invalid essencial file format. Please upload a WAV or MP3 file.")

    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"uploaded_{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(TEST_DATA_DIR, unique_filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Run detection using predict_audio
        prediction, confidence, saved_file_path = predict_audio(file_path, audio_model, audio_processor)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "saved_file_path": saved_file_path
        }
    except Exception as e:
        logger.error(f"Error processing audio file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/image/detect")
async def image_detect(file: UploadFile = File(...)):
    """
    Endpoint for image deepfake detection.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PNG, JPG, or JPEG file.")

    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"image_{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(TEST_DATA_DIR, unique_filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Run detection
        result = detect_image_deepfake(file_path)
        result["saved_file_path"] = file_path
        return result
    except Exception as e:
        logger.error(f"Error processing image file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
