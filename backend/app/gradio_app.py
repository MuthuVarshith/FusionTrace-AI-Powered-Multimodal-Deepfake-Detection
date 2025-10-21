import gradio as gr
import os
import uuid
import cv2
from moviepy.editor import VideoFileClip
import shutil
import time
from app.audio_detection import predict_audio
from app.image_detection import detect_image_deepfake
from app.config import AUDIO_MODEL_PATH, TEST_DATA_DIR, logger
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Global list to store temporary file paths for cleanup
temp_files = []

# Load audio model and processor globally
try:
    audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_PATH)
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL_PATH)
    audio_model.eval()
    logger.info(f"Loaded audio model and processor from {AUDIO_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load audio model or processor: {e}")
    raise

def detect_video_deepfake(video_path, progress=gr.Progress()):
    """
    Process a video file, extract frame and audio, perform late fusion for deepfake detection,
    and return the extracted frame, audio, and frame position info.
    Args:
        video_path (str): Path to the uploaded video file.
        progress: Gradio progress tracker.
    Returns:
        tuple: (detection results, extracted frame path, extracted audio path, frame info, media info)
    """
    global temp_files
    temp_files = []  # Reset temp files list
    progress(0.1, desc="Validating video file...")
    if not video_path or not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
        logger.error("Invalid video file format")
        return {"error": "Please upload a valid MP4, AVI, or MOV video file."}, None, None, None, None

    # Generate unique filenames in TEST_DATA_DIR
    video_unique_filename = f"video_{uuid.uuid4()}{os.path.splitext(video_path)[1]}"
    video_save_path = os.path.join(TEST_DATA_DIR, video_unique_filename)
    frame_path = None
    audio_path = None
    saved_audio_path = None

    try:
        # Copy video to TEST_DATA_DIR to avoid cross-drive issues
        progress(0.2, desc="Copying video file...")
        shutil.copy2(video_path, video_save_path)
        logger.info(f"Copied video to: {video_save_path}")
        temp_files.append(video_save_path)

        # Extract a representative frame
        progress(0.4, desc="Extracting frame...")
        cap = cv2.VideoCapture(video_save_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("Failed to open video file")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                raise RuntimeError("No frames found in video")
            frame_position = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_time = frame_position / fps if fps > 0 else 0
            if not ret:
                raise RuntimeError("Failed to extract frame")
            frame_unique_filename = f"frame_{uuid.uuid4()}.png"
            frame_path = os.path.join(TEST_DATA_DIR, frame_unique_filename)
            cv2.imwrite(frame_path, frame)
            logger.info(f"Saved frame to: {frame_path}")
            temp_files.append(frame_path)
        finally:
            cap.release()

        # Frame info
        frame_info = f"Middle frame at position {frame_position} out of {frame_count} frames (approx. {frame_time:.2f} seconds)"

        # Extract audio
        progress(0.6, desc="Extracting audio...")
        audio_unique_filename = f"audio_{uuid.uuid4()}.wav"
        audio_path = os.path.join(TEST_DATA_DIR, audio_unique_filename)
        try:
            video_clip = VideoFileClip(video_save_path)
            try:
                if video_clip.audio is None:
                    raise RuntimeError("No audio track found in video")
                video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
                logger.info(f"Saved audio to: {audio_path}")
                temp_files.append(audio_path)
            finally:
                video_clip.close()
                if video_clip.audio:
                    video_clip.audio.close()
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio: {str(e)}")

        # Media info
        frame_size = os.path.getsize(frame_path) / 1024 if frame_path and os.path.exists(frame_path) else 0
        audio_size = os.path.getsize(audio_path) / 1024 if audio_path and os.path.exists(audio_path) else 0
        media_info = (
            f"Frame File: {frame_unique_filename} ({frame_size:.2f} KB)\n"
            f"Audio File: {audio_unique_filename} ({audio_size:.2f} KB)"
        )

        # Run individual detections
        progress(0.8, desc="Running image and audio detection...")
        logger.info(f"Processing frame: {frame_path}")
        image_result = detect_image_deepfake(frame_path)
        logger.info(f"Processing audio: {audio_path}")
        audio_prediction, audio_confidence, saved_audio_path = predict_audio(audio_path, audio_model, audio_processor)
        if saved_audio_path and saved_audio_path != audio_path and os.path.exists(saved_audio_path):
            temp_files.append(saved_audio_path)

        # Extract and normalize confidences
        image_prediction = image_result.get("prediction", "Unknown")
        image_confidence_str = image_result.get("confidence", "0.00%")
        try:
            image_confidence = float(image_confidence_str.strip('%')) / 100
        except ValueError:
            logger.error(f"Invalid image confidence format: {image_confidence_str}")
            return {"error": "Invalid image confidence format"}, None, None, None, None

        audio_confidence_str = audio_confidence
        try:
            audio_confidence = float(audio_confidence_str.strip('%')) / 100
        except ValueError:
            logger.error(f"Invalid audio confidence format: {audio_confidence_str}")
            return {"error": "Invalid audio confidence format"}, None, None, None, None

        # Adjust confidences for "Fake" probability
        image_fake_prob = image_confidence if image_prediction == "Fake" else 1 - image_confidence
        audio_fake_prob = audio_confidence if audio_prediction == "Fake" else 1 - audio_confidence

        # Fusion logic: Fake if either image or audio is Fake
        progress(0.9, desc="Fusing predictions...")
        if image_prediction == "Fake" or audio_prediction == "Fake":
            fused_prediction = "Fake"
            fused_confidence = max(image_fake_prob, audio_fake_prob)
        else:
            fused_prediction = "Real"
            fused_confidence = min(1 - image_fake_prob, 1 - audio_fake_prob)

        logger.info(f"Fusion result: prediction={fused_prediction}, confidence={fused_confidence:.2%}, "
                    f"image_prediction={image_prediction}, image_confidence={image_confidence:.2%}, "
                    f"audio_prediction={audio_prediction}, audio_confidence={audio_confidence:.2%}")

        detection_results = {
            "fused_prediction": fused_prediction,
            "fused_confidence": f"{fused_confidence * 100:.2f}%",
            "image_prediction": image_prediction,
            "image_confidence": image_confidence_str,
            "audio_prediction": audio_prediction,
            "audio_confidence": audio_confidence_str
        }

        return detection_results, frame_path, audio_path, frame_info, media_info

    except Exception as e:
        logger.error(f"Error in video detection: {str(e)}")
        return {"error": f"Error in video detection: {str(e)}"}, None, None, None, None

    finally:
        # Clean up video file only
        try:
            if os.path.exists(video_save_path):
                for attempt in range(3):
                    try:
                        os.remove(video_save_path)
                        logger.info(f"Deleted video file: {video_save_path}")
                        temp_files.remove(video_save_path) if video_save_path in temp_files else None
                        break
                    except PermissionError as e:
                        logger.warning(f"Video deletion attempt {attempt + 1} failed: {e}")
                        time.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to delete video file {video_save_path}: {e}")

def cleanup_files():
    """Clean up all temporary files stored in temp_files."""
    global temp_files
    deleted_files = []
    for file_path in temp_files[:]:  # Copy to avoid modifying during iteration
        try:
            if file_path and os.path.exists(file_path):
                for attempt in range(3):
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted file: {file_path}")
                        deleted_files.append(file_path)
                        break
                    except PermissionError as e:
                        logger.warning(f"File deletion attempt {attempt + 1} failed: {e}")
                        time.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")
    
    # Update temp_files
    temp_files = [f for f in temp_files if f not in deleted_files]
    return "All temporary files have been deleted." if deleted_files else "No temporary files to delete."

def create_gradio_interface():
    """Create and return the Gradio interface for multimodal deepfake detection."""
    css = """
    .gradio-container {max-width: 1200px; margin: auto; font-family: 'Helvetica', sans-serif; background: #f4f7fa;}
    h1 {color: #1a3c66; text-align: center; margin-bottom: 20px;}
    .section {border: 1px solid #d1d9e6; border-radius: 10px; padding: 20px; background: #ffffff; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .output-json {background: #e6f3ff; padding: 15px; border-radius: 8px; font-size: 14px;}
    .output-image, .output-audio {border: 1px solid #d1d9e6; border-radius: 8px; padding: 10px; background: #f9f9f9;}
    .output-text {font-size: 14px; color: #34495e;}
    button {background-color: #1a3c66 !important; color: white !important; border-radius: 8px; padding: 10px 20px; font-size: 16px;}
    button:hover {background-color: #15325a !important;}
    .progress-text {color: #7f8c8d; font-style: italic;}
    label {font-weight: bold; color: #1a3c66;}
    """

    with gr.Blocks(title="Multimodal Deepfake Detection", css=css) as demo:
        gr.Markdown("""
        # Multimodal Deepfake Detection
        Upload a video to detect deepfakes using AI-powered image and audio analysis. Results are displayed below with extracted media and detailed info.
        """, elem_classes="section")

        # Input Section
        with gr.Group(elem_classes="section"):
            gr.Markdown("### Upload Video")
            video_input = gr.Video(label="Select Video (MP4, AVI, MOV)", format="mp4")
            submit_button = gr.Button("Analyze Video", variant="primary")
            gr.Markdown("**Instructions**: Upload a video and click 'Analyze Video'. View results, frame, and audio below.")

        # Results Section
        with gr.Group(elem_classes="section"):
            gr.Markdown("### Detection Results")
            with gr.Row():
                detection_output = gr.JSON(label="Prediction Results", elem_classes="output-json")
                with gr.Column():
                    frame_output = gr.Image(label="Extracted Frame", elem_classes="output-image")
                    audio_output = gr.Audio(label="Extracted Audio", elem_classes="output-audio")
            
            with gr.Row():
                frame_info_output = gr.Textbox(label="Frame Position Info", elem_classes="output-text")
                media_info_output = gr.Textbox(label="Media File Info", elem_classes="output-text")

        # Cleanup Section
        with gr.Group(elem_classes="section"):
            gr.Markdown("### Cleanup")
            cleanup_button = gr.Button("Clean Up Temporary Files")
            cleanup_output = gr.Textbox(label="Cleanup Status", elem_classes="output-text")
            gr.Markdown("**Note**: Click to delete extracted frame and audio files. Video files are deleted automatically after analysis.")

        # About Section
        with gr.Group(elem_classes="section"):
            gr.Markdown("""
            ### About This App
            - **Image Analysis**: Uses EfficientNetV2 on the middle frame of the video.
            - **Audio Analysis**: Uses Wav2Vec2 on the extracted audio.
            - **Fusion**: Predicts 'Fake' if either image or audio is fake; otherwise, 'Real'.
            - **Privacy**: Video files are deleted immediately; frame and audio files are deleted after cleanup.
            """)

        submit_button.click(
            fn=detect_video_deepfake,
            inputs=video_input,
            outputs=[detection_output, frame_output, audio_output, frame_info_output, media_info_output]
        )
        
        cleanup_button.click(
            fn=cleanup_files,
            inputs=[],
            outputs=cleanup_output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)