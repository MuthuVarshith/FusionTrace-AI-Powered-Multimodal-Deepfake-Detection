"""
Video deepfake detection module with frame-by-frame analysis and temporal consistency.
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Dict, List, Optional
from .image_detection import detect_image_deepfake
from .audio_detection import predict_audio
from .enhanced_image_detection import detect_enhanced_image
from .config import TEST_DATA_DIR, logger

class VideoProcessor:
    """Video processing utilities for deepfake detection."""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    def extract_frames(self, video_path: str, max_frames: int = 30, 
                     frame_interval: int = 1) -> List[np.ndarray]:
        """Extract frames from video for analysis."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
            
            # Calculate frame sampling
            if total_frames <= max_frames:
                frame_indices = list(range(0, total_frames, max(1, frame_interval)))
            else:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    logger.warning(f"Failed to read frame {frame_idx}")
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video for analysis."""
        try:
            import subprocess
            
            # Create temporary audio file
            audio_filename = f"temp_audio_{os.path.basename(video_path)}.wav"
            audio_path = os.path.join(TEST_DATA_DIR, audio_filename)
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', 
                '-ac', '1', '-y', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Audio extracted to: {audio_path}")
                return audio_path
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None
    
    def get_video_metadata(self, video_path: str) -> Dict:
        """Get video metadata."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}
            
            metadata = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                'codec': cap.get(cv2.CAP_PROP_FOURCC)
            }
            
            cap.release()
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting video metadata: {e}")
            return {}

class TemporalConsistencyAnalyzer:
    """Analyze temporal consistency across video frames."""
    
    def __init__(self):
        self.consistency_threshold = 0.7
    
    def analyze_temporal_consistency(self, frame_predictions: List[Dict]) -> Dict:
        """Analyze temporal consistency of predictions across frames."""
        try:
            if not frame_predictions:
                return {'consistency_score': 0.0, 'temporal_anomalies': []}
            
            # Extract predictions and confidences
            predictions = [fp.get('prediction', 'Unknown') for fp in frame_predictions]
            confidences = [float(fp.get('confidence', '0%').replace('%', '')) / 100.0 for fp in frame_predictions]
            
            # Calculate consistency metrics
            fake_ratio = sum(1 for p in predictions if p.lower() == 'fake') / len(predictions)
            confidence_variance = np.var(confidences)
            confidence_mean = np.mean(confidences)
            
            # Detect temporal anomalies
            temporal_anomalies = []
            for i in range(1, len(predictions)):
                if predictions[i] != predictions[i-1]:
                    temporal_anomalies.append({
                        'frame': i,
                        'transition': f"{predictions[i-1]} → {predictions[i]}",
                        'confidence_change': abs(confidences[i] - confidences[i-1])
                    })
            
            # Calculate overall consistency score
            consistency_score = 1.0 - (confidence_variance + len(temporal_anomalies) * 0.1)
            consistency_score = max(0.0, min(1.0, consistency_score))
            
            return {
                'consistency_score': consistency_score,
                'fake_ratio': fake_ratio,
                'confidence_mean': confidence_mean,
                'confidence_variance': confidence_variance,
                'temporal_anomalies': temporal_anomalies,
                'frame_count': len(frame_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error in temporal consistency analysis: {e}")
            return {'consistency_score': 0.0, 'temporal_anomalies': []}

class LateFusionDetector:
    """Late fusion logic for combining multiple modalities."""
    
    def __init__(self):
        self.modality_weights = {
            'visual': 0.6,
            'audio': 0.3,
            'temporal': 0.1
        }
    
    def fuse_predictions(self, visual_results: Dict, audio_results: Optional[Dict] = None, 
                        temporal_results: Optional[Dict] = None) -> Dict:
        """Fuse predictions from multiple modalities using late fusion."""
        try:
            # Extract visual prediction
            visual_pred = visual_results.get('prediction', 'Unknown')
            visual_conf = float(visual_results.get('confidence', '0%').replace('%', '')) / 100.0
            
            # Initialize fusion scores
            fake_score = 0.0
            real_score = 0.0
            modality_contributions = {}
            
            # Visual modality contribution
            if visual_pred.lower() == 'fake':
                fake_score += self.modality_weights['visual'] * visual_conf
            else:
                real_score += self.modality_weights['visual'] * visual_conf
            
            modality_contributions['visual'] = {
                'prediction': visual_pred,
                'confidence': visual_conf,
                'weight': self.modality_weights['visual']
            }
            
            # Audio modality contribution (if available)
            if audio_results:
                audio_pred = audio_results.get('prediction', 'Unknown')
                audio_conf = float(audio_results.get('confidence', '0%').replace('%', '')) / 100.0
                
                if audio_pred.lower() == 'fake':
                    fake_score += self.modality_weights['audio'] * audio_conf
                else:
                    real_score += self.modality_weights['audio'] * audio_conf
                
                modality_contributions['audio'] = {
                    'prediction': audio_pred,
                    'confidence': audio_conf,
                    'weight': self.modality_weights['audio']
                }
            
            # Temporal modality contribution (if available)
            if temporal_results:
                temporal_score = temporal_results.get('consistency_score', 0.5)
                # Lower consistency suggests more manipulation
                temporal_fake_score = 1.0 - temporal_score
                
                fake_score += self.modality_weights['temporal'] * temporal_fake_score
                real_score += self.modality_weights['temporal'] * temporal_score
                
                modality_contributions['temporal'] = {
                    'consistency_score': temporal_score,
                    'weight': self.modality_weights['temporal']
                }
            
            # Determine final prediction
            final_prediction = 'Fake' if fake_score > real_score else 'Real'
            final_confidence = max(fake_score, real_score)
            
            return {
                'prediction': final_prediction,
                'confidence': f"{final_confidence * 100:.2f}%",
                'fake_score': fake_score,
                'real_score': real_score,
                'modality_contributions': modality_contributions,
                'fusion_method': 'late_fusion'
            }
            
        except Exception as e:
            logger.error(f"Error in late fusion: {e}")
            return {
                'prediction': 'Unknown',
                'confidence': '0.00%',
                'error': str(e)
            }

class VideoDeepfakeDetector:
    """Main video deepfake detection class."""
    
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.temporal_analyzer = TemporalConsistencyAnalyzer()
        self.late_fusion = LateFusionDetector()
    
    def detect_video_deepfake(self, video_path: str, enhanced: bool = True) -> Dict:
        """Detect deepfakes in video using multimodal analysis."""
        try:
            logger.info(f"Starting video deepfake detection: {video_path}")
            
            # Get video metadata
            metadata = self.video_processor.get_video_metadata(video_path)
            logger.info(f"Video metadata: {metadata}")
            
            # Extract frames for analysis
            frames = self.video_processor.extract_frames(video_path, max_frames=30)
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Analyze each frame
            frame_results = []
            for i, frame in enumerate(frames):
                try:
                    # Save frame temporarily
                    frame_path = os.path.join(TEST_DATA_DIR, f"frame_{i}_{os.path.basename(video_path)}.jpg")
                    frame_img = Image.fromarray(frame)
                    frame_img.save(frame_path)
                    
                    # Analyze frame
                    if enhanced:
                        frame_result = detect_enhanced_image(frame_path)
                    else:
                        frame_result = detect_image_deepfake(frame_path)
                    
                    frame_results.append({
                        'frame_index': i,
                        'prediction': frame_result.get('prediction', 'Unknown'),
                        'confidence': frame_result.get('confidence', '0%'),
                        'enhanced_data': frame_result if enhanced else None
                    })
                    
                    # Clean up temporary frame
                    os.remove(frame_path)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing frame {i}: {e}")
                    frame_results.append({
                        'frame_index': i,
                        'prediction': 'Error',
                        'confidence': '0%',
                        'error': str(e)
                    })
            
            # Extract audio for analysis
            audio_path = self.video_processor.extract_audio(video_path)
            audio_results = None
            if audio_path:
                try:
                    from .config import audio_model, audio_processor
                    audio_pred, audio_conf, _ = predict_audio(audio_path, audio_model, audio_processor)
                    audio_results = {
                        'prediction': audio_pred,
                        'confidence': audio_conf
                    }
                    # Clean up temporary audio
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Error analyzing audio: {e}")
            
            # Analyze temporal consistency
            temporal_results = self.temporal_analyzer.analyze_temporal_consistency(frame_results)
            
            # Get representative frame result for visual analysis
            visual_results = frame_results[0] if frame_results else {'prediction': 'Unknown', 'confidence': '0%'}
            
            # Apply late fusion
            fusion_results = self.late_fusion.fuse_predictions(
                visual_results, audio_results, temporal_results
            )
            
            # Calculate video-level statistics
            fake_frames = sum(1 for fr in frame_results if fr.get('prediction', '').lower() == 'fake')
            total_frames = len(frame_results)
            fake_ratio = fake_frames / total_frames if total_frames > 0 else 0
            
            return {
                'prediction': fusion_results['prediction'],
                'confidence': fusion_results['confidence'],
                'video_metadata': metadata,
                'frame_analysis': {
                    'total_frames': total_frames,
                    'fake_frames': fake_frames,
                    'fake_ratio': fake_ratio,
                    'frame_results': frame_results
                },
                'audio_analysis': audio_results,
                'temporal_analysis': temporal_results,
                'fusion_analysis': fusion_results,
                'enhanced_analysis': enhanced
            }
            
        except Exception as e:
            logger.error(f"Error in video deepfake detection: {e}")
            return {
                'prediction': 'Error',
                'confidence': '0.00%',
                'error': str(e)
            }

# Global instance
video_detector = None

def get_video_detector():
    """Get or create video detector instance."""
    global video_detector
    if video_detector is None:
        video_detector = VideoDeepfakeDetector()
    return video_detector

def detect_video_deepfake(video_path: str, enhanced: bool = True) -> Dict:
    """Convenience function for video deepfake detection."""
    detector = get_video_detector()
    return detector.detect_video_deepfake(video_path, enhanced)
