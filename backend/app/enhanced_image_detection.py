"""
Enhanced image detection module with Grad-CAM++, fake intensity, and forgery type classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from timm import create_model
import logging
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import io
import base64
from .config import IMAGE_MODEL_PATH

logger = logging.getLogger(__name__)

class GradCAM:
    """Grad-CAM implementation for visualizing important regions in images."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap."""
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()

class GradCAMPlusPlus(GradCAM):
    """Enhanced Grad-CAM++ implementation."""
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM++ heatmap."""
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM++
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Calculate alpha values (Grad-CAM++)
        alpha = torch.zeros_like(gradients)
        for i in range(gradients.shape[0]):
            for j in range(gradients.shape[1]):
                for k in range(gradients.shape[2]):
                    alpha[i, j, k] = gradients[i, j, k] / (2 * activations[i, j, k] + 
                                                          torch.sum(gradients[i, :, :] * activations[i, :, :]))
        
        # Calculate weights
        weights = torch.sum(alpha * F.relu(gradients), dim=(1, 2))
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()

class ForgeryTypeClassifier:
    """Classifier for different types of image forgery."""
    
    def __init__(self):
        self.forgery_types = {
            'face_swap': 'Face Swap',
            'deepfake': 'Deepfake',
            'face_reenactment': 'Face Reenactment',
            'face_synthesis': 'Face Synthesis',
            'inpainting': 'Inpainting',
            'splicing': 'Splicing',
            'authentic': 'Authentic'
        }
    
    def classify_forgery_type(self, image_tensor, prediction, confidence):
        """Classify the type of forgery based on image characteristics."""
        # This is a simplified implementation
        # In practice, you would train a separate classifier for forgery types
        
        if prediction.lower() == 'real':
            return {
                'forgery_type': 'authentic',
                'forgery_type_name': 'Authentic',
                'confidence': confidence
            }
        
        # Analyze image characteristics to determine forgery type
        # This is a heuristic approach - in practice, use a trained classifier
        
        # Convert tensor to numpy for analysis
        if len(image_tensor.shape) == 4:
            image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        image_np = (image_np + 1) / 2  # Denormalize
        image_np = np.clip(image_np, 0, 1)
        
        # Convert to OpenCV format
        image_cv = (image_np * 255).astype(np.uint8)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Analyze face region characteristics
        forgery_type = self._analyze_face_characteristics(image_cv, confidence)
        
        return {
            'forgery_type': forgery_type,
            'forgery_type_name': self.forgery_types.get(forgery_type, 'Unknown'),
            'confidence': confidence
        }
    
    def _analyze_face_characteristics(self, image_cv, confidence):
        """Analyze face characteristics to determine forgery type."""
        # Simplified heuristic analysis
        # In practice, use computer vision techniques and ML models
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return 'splicing'  # No face detected, likely splicing
        
        # Analyze face region
        x, y, w, h = faces[0]
        face_roi = image_cv[y:y+h, x:x+w]
        
        # Calculate image statistics
        mean_intensity = np.mean(face_roi)
        std_intensity = np.std(face_roi)
        
        # Heuristic classification based on confidence and characteristics
        if confidence > 0.9:
            if std_intensity < 30:  # Low variation
                return 'deepfake'
            else:
                return 'face_swap'
        elif confidence > 0.7:
            return 'face_reenactment'
        else:
            return 'face_synthesis'

class FakeIntensityAnalyzer:
    """Analyzer for fake intensity scoring."""
    
    def __init__(self):
        self.intensity_levels = {
            'low': (0.0, 0.3),
            'medium': (0.3, 0.7),
            'high': (0.7, 1.0)
        }
    
    def calculate_fake_intensity(self, prediction, confidence, forgery_type=None):
        """Calculate fake intensity score."""
        if prediction.lower() == 'real':
            return {
                'intensity_score': 0.0,
                'intensity_level': 'none',
                'intensity_description': 'Authentic content'
            }
        
        # Base intensity from confidence
        base_intensity = float(confidence.replace('%', '')) / 100.0
        
        # Adjust based on forgery type
        type_multipliers = {
            'deepfake': 1.0,
            'face_swap': 0.9,
            'face_reenactment': 0.8,
            'face_synthesis': 0.7,
            'inpainting': 0.6,
            'splicing': 0.5
        }
        
        if forgery_type and forgery_type in type_multipliers:
            adjusted_intensity = base_intensity * type_multipliers[forgery_type]
        else:
            adjusted_intensity = base_intensity
        
        # Determine intensity level
        if adjusted_intensity < 0.3:
            level = 'low'
        elif adjusted_intensity < 0.7:
            level = 'medium'
        else:
            level = 'high'
        
        # Generate description
        descriptions = {
            'low': 'Subtle manipulation detected',
            'medium': 'Moderate manipulation detected',
            'high': 'Severe manipulation detected'
        }
        
        return {
            'intensity_score': adjusted_intensity,
            'intensity_level': level,
            'intensity_description': descriptions.get(level, 'Unknown intensity')
        }

class EnhancedImageDetector:
    """Enhanced image detector with Grad-CAM++, fake intensity, and forgery classification."""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        self.gradcam = None
        self.forgery_classifier = ForgeryTypeClassifier()
        self.intensity_analyzer = FakeIntensityAnalyzer()
        
        # Setup Grad-CAM++ (use the last convolutional layer)
        self._setup_gradcam()
    
    def _load_model(self, model_path):
        """Load the trained model."""
        try:
            from .image_detection import EfficientNetV2
            model = EfficientNetV2()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            logger.info(f"Loaded enhanced image model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_transform(self):
        """Get image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    
    def _setup_gradcam(self):
        """Setup Grad-CAM++ for the model."""
        try:
            # Find the last convolutional layer
            target_layer = None
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
            
            if target_layer:
                self.gradcam = GradCAMPlusPlus(self.model, target_layer)
                logger.info("Grad-CAM++ setup completed")
            else:
                logger.warning("No convolutional layer found for Grad-CAM++")
        except Exception as e:
            logger.error(f"Failed to setup Grad-CAM++: {e}")
    
    def detect_enhanced(self, image_path: str) -> Dict:
        """Perform enhanced deepfake detection with all features."""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Basic prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                prob = torch.sigmoid(output).item()
                prediction = "Fake" if prob >= 0.5 else "Real"
                confidence = prob if prediction == "Fake" else 1 - prob
            
            # Generate Grad-CAM++ visualization
            gradcam_heatmap = None
            gradcam_overlay = None
            if self.gradcam:
                try:
                    gradcam_heatmap = self.gradcam.generate_cam(img_tensor)
                    gradcam_overlay = self._create_gradcam_overlay(img, gradcam_heatmap)
                except Exception as e:
                    logger.warning(f"Grad-CAM++ generation failed: {e}")
            
            # Classify forgery type
            forgery_analysis = self.forgery_classifier.classify_forgery_type(
                img_tensor, prediction, confidence
            )
            
            # Calculate fake intensity
            intensity_analysis = self.intensity_analyzer.calculate_fake_intensity(
                prediction, f"{confidence * 100:.2f}%", 
                forgery_analysis.get('forgery_type')
            )
            
            return {
                "prediction": prediction,
                "confidence": f"{confidence * 100:.2f}%",
                "gradcam_heatmap": gradcam_heatmap,
                "gradcam_overlay": gradcam_overlay,
                "forgery_type": forgery_analysis.get('forgery_type'),
                "forgery_type_name": forgery_analysis.get('forgery_type_name'),
                "fake_intensity": intensity_analysis.get('intensity_score'),
                "intensity_level": intensity_analysis.get('intensity_level'),
                "intensity_description": intensity_analysis.get('intensity_description'),
                "analysis_details": {
                    "gradcam_available": gradcam_heatmap is not None,
                    "forgery_classification": forgery_analysis,
                    "intensity_analysis": intensity_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced detection: {e}")
            return {
                "prediction": "Error",
                "confidence": "0.00%",
                "error": str(e)
            }
    
    def _create_gradcam_overlay(self, original_img, heatmap):
        """Create Grad-CAM++ overlay visualization."""
        try:
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap, original_img.size)
            
            # Create colormap
            colormap = cm.get_cmap('jet')
            heatmap_colored = colormap(heatmap_resized)[:, :, :3]
            
            # Convert original image to numpy
            img_np = np.array(original_img)
            img_np = img_np / 255.0
            
            # Create overlay
            overlay = 0.6 * heatmap_colored + 0.4 * img_np
            overlay = np.clip(overlay, 0, 1)
            
            # Convert to PIL Image
            overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
            
            # Convert to base64 for web display
            buffer = io.BytesIO()
            overlay_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error creating Grad-CAM++ overlay: {e}")
            return None

# Global instance
enhanced_detector = None

def get_enhanced_detector():
    """Get or create enhanced detector instance."""
    global enhanced_detector
    if enhanced_detector is None:
        enhanced_detector = EnhancedImageDetector(IMAGE_MODEL_PATH)
    return enhanced_detector

def detect_enhanced_image(image_path: str) -> Dict:
    """Convenience function for enhanced image detection."""
    detector = get_enhanced_detector()
    return detector.detect_enhanced(image_path)
