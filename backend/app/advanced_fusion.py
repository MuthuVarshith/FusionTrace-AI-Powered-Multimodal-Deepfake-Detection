"""
Advanced fusion mechanisms for multimodal deepfake detection.
Includes intelligent weighted fusion, late fusion classifier, and enhanced explainability.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

logger = logging.getLogger(__name__)

class IntelligentWeightedFusion:
    """Intelligent weighted fusion with conditional logic and confidence-based weighting."""
    
    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        self.modality_weights = {
            'visual': 0.5,
            'audio': 0.3,
            'temporal': 0.2
        }
    
    def fuse_with_confidence_logic(self, visual_score: float, audio_score: float, 
                                 temporal_score: float, visual_conf: float, 
                                 audio_conf: float, temporal_conf: float) -> Dict:
        """Apply intelligent fusion based on confidence levels and conditional logic."""
        try:
            # Determine confidence levels
            visual_level = self._get_confidence_level(visual_conf)
            audio_level = self._get_confidence_level(audio_conf)
            temporal_level = self._get_confidence_level(temporal_conf)
            
            # Apply conditional logic for weighting
            weights = self._calculate_adaptive_weights(
                visual_score, audio_score, temporal_score,
                visual_conf, audio_conf, temporal_conf,
                visual_level, audio_level, temporal_level
            )
            
            # Calculate weighted fusion
            fused_score = (
                weights['visual'] * visual_score +
                weights['audio'] * audio_score +
                weights['temporal'] * temporal_score
            )
            
            # Determine prediction
            prediction = 'Fake' if fused_score > 0.5 else 'Real'
            confidence = max(fused_score, 1 - fused_score)
            
            # Calculate fusion confidence
            fusion_confidence = self._calculate_fusion_confidence(
                visual_conf, audio_conf, temporal_conf, weights
            )
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'fused_score': fused_score,
                'fusion_confidence': fusion_confidence,
                'weights': weights,
                'confidence_levels': {
                    'visual': visual_level,
                    'audio': audio_level,
                    'temporal': temporal_level
                },
                'fusion_method': 'intelligent_weighted',
                'explanation': self._generate_fusion_explanation(
                    weights, visual_level, audio_level, temporal_level
                )
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent fusion: {e}")
            return self._fallback_fusion(visual_score, audio_score, temporal_score)
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Determine confidence level based on threshold."""
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_adaptive_weights(self, visual_score: float, audio_score: float,
                                  temporal_score: float, visual_conf: float,
                                  audio_conf: float, temporal_conf: float,
                                  visual_level: str, audio_level: str, 
                                  temporal_level: str) -> Dict:
        """Calculate adaptive weights based on confidence and agreement."""
        base_weights = self.modality_weights.copy()
        
        # High confidence gets more weight
        if visual_level == 'high':
            base_weights['visual'] *= 1.5
        if audio_level == 'high':
            base_weights['audio'] *= 1.5
        if temporal_level == 'high':
            base_weights['temporal'] *= 1.5
        
        # Agreement bonus: if two modalities agree, boost their weight
        if abs(visual_score - audio_score) < 0.2:  # Visual and audio agree
            base_weights['visual'] *= 1.2
            base_weights['audio'] *= 1.2
        if abs(visual_score - temporal_score) < 0.2:  # Visual and temporal agree
            base_weights['visual'] *= 1.2
            base_weights['temporal'] *= 1.2
        if abs(audio_score - temporal_score) < 0.2:  # Audio and temporal agree
            base_weights['audio'] *= 1.2
            base_weights['temporal'] *= 1.2
        
        # Disagreement penalty: if modalities strongly disagree, reduce weight
        if abs(visual_score - audio_score) > 0.6:  # Strong disagreement
            base_weights['visual'] *= 0.8
            base_weights['audio'] *= 0.8
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}
    
    def _calculate_fusion_confidence(self, visual_conf: float, audio_conf: float,
                                   temporal_conf: float, weights: Dict) -> float:
        """Calculate overall fusion confidence."""
        weighted_confidence = (
            weights['visual'] * visual_conf +
            weights['audio'] * audio_conf +
            weights['temporal'] * temporal_conf
        )
        return weighted_confidence
    
    def _generate_fusion_explanation(self, weights: Dict, visual_level: str,
                                   audio_level: str, temporal_level: str) -> str:
        """Generate human-readable explanation of fusion process."""
        dominant_modality = max(weights, key=weights.get)
        explanation = f"Fusion analysis: {dominant_modality.title()} modality weighted highest ({weights[dominant_modality]:.1%})"
        
        if visual_level == 'high' and audio_level == 'high':
            explanation += " with high confidence from both visual and audio analysis."
        elif visual_level == 'high' or audio_level == 'high':
            explanation += f" with high confidence from {visual_level if visual_level == 'high' else audio_level} analysis."
        else:
            explanation += " with moderate confidence across modalities."
        
        return explanation
    
    def _fallback_fusion(self, visual_score: float, audio_score: float, 
                        temporal_score: float) -> Dict:
        """Fallback to simple averaging if intelligent fusion fails."""
        avg_score = (visual_score + audio_score + temporal_score) / 3
        return {
            'prediction': 'Fake' if avg_score > 0.5 else 'Real',
            'confidence': max(avg_score, 1 - avg_score),
            'fused_score': avg_score,
            'fusion_method': 'simple_average',
            'explanation': 'Simple averaging due to fusion error'
        }

class LateFusionClassifier:
    """Late fusion classifier using machine learning models."""
    
    def __init__(self, model_path: str = "models/late_fusion_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'visual_score', 'audio_score', 'temporal_score',
            'visual_confidence', 'audio_confidence', 'temporal_confidence',
            'visual_audio_agreement', 'visual_temporal_agreement', 'audio_temporal_agreement',
            'score_variance', 'confidence_variance'
        ]
        self._load_or_train_model()
    
    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded late fusion model from {self.model_path}")
            else:
                logger.info("No existing model found, will train on first use")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def train_model(self, training_data: List[Dict]):
        """Train the late fusion classifier."""
        try:
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Late fusion model trained with accuracy: {accuracy:.3f}")
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            
            return {
                'accuracy': accuracy,
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the classifier."""
        features = []
        labels = []
        
        for data in training_data:
            # Extract features
            feature_vector = [
                data.get('visual_score', 0.5),
                data.get('audio_score', 0.5),
                data.get('temporal_score', 0.5),
                data.get('visual_confidence', 0.5),
                data.get('audio_confidence', 0.5),
                data.get('temporal_confidence', 0.5),
                data.get('visual_audio_agreement', 0.5),
                data.get('visual_temporal_agreement', 0.5),
                data.get('audio_temporal_agreement', 0.5),
                data.get('score_variance', 0.1),
                data.get('confidence_variance', 0.1)
            ]
            
            features.append(feature_vector)
            labels.append(1 if data.get('ground_truth', 'fake').lower() == 'fake' else 0)
        
        return np.array(features), np.array(labels)
    
    def predict(self, visual_score: float, audio_score: float, temporal_score: float,
               visual_conf: float, audio_conf: float, temporal_conf: float) -> Dict:
        """Predict using the trained late fusion classifier."""
        try:
            if self.model is None:
                # Fallback to intelligent weighted fusion
                intelligent_fusion = IntelligentWeightedFusion()
                return intelligent_fusion.fuse_with_confidence_logic(
                    visual_score, audio_score, temporal_score,
                    visual_conf, audio_conf, temporal_conf
                )
            
            # Prepare features
            features = np.array([[
                visual_score, audio_score, temporal_score,
                visual_conf, audio_conf, temporal_conf,
                self._calculate_agreement(visual_score, audio_score),
                self._calculate_agreement(visual_score, temporal_score),
                self._calculate_agreement(audio_score, temporal_score),
                self._calculate_variance([visual_score, audio_score, temporal_score]),
                self._calculate_variance([visual_conf, audio_conf, temporal_conf])
            ]])
            
            # Predict
            prediction_proba = self.model.predict_proba(features)[0]
            fake_probability = prediction_proba[1]
            
            return {
                'prediction': 'Fake' if fake_probability > 0.5 else 'Real',
                'confidence': max(fake_probability, 1 - fake_probability),
                'fused_score': fake_probability,
                'fusion_method': 'late_fusion_classifier',
                'model_confidence': fake_probability,
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Error in late fusion prediction: {e}")
            # Fallback to intelligent fusion
            intelligent_fusion = IntelligentWeightedFusion()
            return intelligent_fusion.fuse_with_confidence_logic(
                visual_score, audio_score, temporal_score,
                visual_conf, audio_conf, temporal_conf
            )
    
    def _calculate_agreement(self, score1: float, score2: float) -> float:
        """Calculate agreement between two scores."""
        return 1.0 - abs(score1 - score2)
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores."""
        return np.var(scores)

class EnhancedForgeryClassifier:
    """Enhanced forgery type classifier with specific categories."""
    
    def __init__(self):
        self.forgery_types = {
            'face_swap': {
                'name': 'Face Swap',
                'description': 'Traditional face swapping between individuals',
                'indicators': ['face_boundary_inconsistency', 'lighting_mismatch', 'resolution_difference']
            },
            'lip_sync': {
                'name': 'Lip Sync Manipulation',
                'description': 'Audio-visual synchronization manipulation',
                'indicators': ['lip_mouth_mismatch', 'audio_visual_desync', 'temporal_inconsistency']
            },
            'audio_clone': {
                'name': 'Audio Clone',
                'description': 'AI-generated voice synthesis',
                'indicators': ['voice_synthesis_markers', 'spectral_anomalies', 'prosody_inconsistency']
            },
            'deepfake': {
                'name': 'Deepfake',
                'description': 'AI-generated face replacement',
                'indicators': ['ai_generation_markers', 'face_consistency', 'temporal_artifacts']
            },
            'mixed': {
                'name': 'Mixed Manipulation',
                'description': 'Combination of multiple manipulation techniques',
                'indicators': ['multiple_techniques', 'complex_artifacts', 'cross_modal_inconsistency']
            },
            'authentic': {
                'name': 'Authentic',
                'description': 'Genuine, unmanipulated content',
                'indicators': ['natural_consistency', 'no_artifacts', 'cross_modal_alignment']
            }
        }
    
    def classify_forgery_type(self, visual_analysis: Dict, audio_analysis: Dict, 
                           temporal_analysis: Dict, fusion_analysis: Dict) -> Dict:
        """Classify the specific type of forgery based on comprehensive analysis."""
        try:
            # Extract key indicators
            indicators = self._extract_indicators(visual_analysis, audio_analysis, 
                                               temporal_analysis, fusion_analysis)
            
            # Score each forgery type
            type_scores = {}
            for forgery_type, config in self.forgery_types.items():
                if forgery_type == 'authentic':
                    continue
                
                score = self._calculate_type_score(indicators, config['indicators'])
                type_scores[forgery_type] = score
            
            # Determine best match
            best_type = max(type_scores, key=type_scores.get)
            best_score = type_scores[best_type]
            
            # Check if authentic
            authentic_score = self._calculate_authentic_score(indicators)
            if authentic_score > best_score:
                best_type = 'authentic'
                best_score = authentic_score
            
            return {
                'forgery_type': best_type,
                'forgery_type_name': self.forgery_types[best_type]['name'],
                'confidence': best_score,
                'description': self.forgery_types[best_type]['description'],
                'detected_indicators': self._get_detected_indicators(indicators),
                'type_scores': type_scores
            }
            
        except Exception as e:
            logger.error(f"Error in forgery classification: {e}")
            return {
                'forgery_type': 'unknown',
                'forgery_type_name': 'Unknown',
                'confidence': 0.0,
                'description': 'Unable to classify forgery type'
            }
    
    def _extract_indicators(self, visual_analysis: Dict, audio_analysis: Dict,
                          temporal_analysis: Dict, fusion_analysis: Dict) -> Dict:
        """Extract relevant indicators from analysis results."""
        indicators = {}
        
        # Visual indicators
        if 'gradcam_available' in visual_analysis:
            indicators['attention_focus'] = visual_analysis.get('gradcam_available', False)
        
        if 'forgery_type' in visual_analysis:
            indicators['visual_forgery_type'] = visual_analysis.get('forgery_type', 'unknown')
        
        if 'fake_intensity' in visual_analysis:
            indicators['manipulation_intensity'] = visual_analysis.get('fake_intensity', 0)
        
        # Audio indicators
        if audio_analysis:
            indicators['audio_confidence'] = float(audio_analysis.get('confidence', '0%').replace('%', '')) / 100.0
            indicators['audio_prediction'] = audio_analysis.get('prediction', 'unknown')
        
        # Temporal indicators
        if 'consistency_score' in temporal_analysis:
            indicators['temporal_consistency'] = temporal_analysis.get('consistency_score', 0.5)
        
        if 'temporal_anomalies' in temporal_analysis:
            indicators['temporal_anomaly_count'] = len(temporal_analysis.get('temporal_anomalies', []))
        
        # Fusion indicators
        if 'modality_contributions' in fusion_analysis:
            contributions = fusion_analysis.get('modality_contributions', {})
            indicators['modality_agreement'] = self._calculate_modality_agreement(contributions)
        
        return indicators
    
    def _calculate_type_score(self, indicators: Dict, type_indicators: List[str]) -> float:
        """Calculate score for a specific forgery type."""
        score = 0.0
        total_weight = 0.0
        
        for indicator in type_indicators:
            if indicator in indicators:
                weight = 1.0
                if indicator == 'manipulation_intensity':
                    weight = 2.0  # Higher weight for intensity
                elif indicator == 'temporal_consistency':
                    weight = 1.5  # Higher weight for temporal analysis
                
                score += indicators[indicator] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_authentic_score(self, indicators: Dict) -> float:
        """Calculate score for authentic content."""
        authentic_indicators = ['natural_consistency', 'no_artifacts', 'cross_modal_alignment']
        return self._calculate_type_score(indicators, authentic_indicators)
    
    def _calculate_modality_agreement(self, contributions: Dict) -> float:
        """Calculate agreement between different modalities."""
        if len(contributions) < 2:
            return 1.0
        
        predictions = [contrib.get('prediction', 'unknown') for contrib in contributions.values()]
        return len(set(predictions)) == 1  # All predictions are the same
    
    def _get_detected_indicators(self, indicators: Dict) -> List[str]:
        """Get list of detected indicators."""
        detected = []
        for indicator, value in indicators.items():
            if isinstance(value, bool) and value:
                detected.append(indicator)
            elif isinstance(value, (int, float)) and value > 0.5:
                detected.append(f"{indicator}: {value:.2f}")
        return detected

# Global instances
intelligent_fusion = IntelligentWeightedFusion()
late_fusion_classifier = LateFusionClassifier()
enhanced_forgery_classifier = EnhancedForgeryClassifier()

def advanced_fusion_predict(visual_score: float, audio_score: float, temporal_score: float,
                          visual_conf: float, audio_conf: float, temporal_conf: float,
                          use_classifier: bool = True) -> Dict:
    """Advanced fusion prediction using intelligent weighting and/or ML classifier."""
    if use_classifier:
        return late_fusion_classifier.predict(
            visual_score, audio_score, temporal_score,
            visual_conf, audio_conf, temporal_conf
        )
    else:
        return intelligent_fusion.fuse_with_confidence_logic(
            visual_score, audio_score, temporal_score,
            visual_conf, audio_conf, temporal_conf
        )

def classify_enhanced_forgery_type(visual_analysis: Dict, audio_analysis: Dict,
                                 temporal_analysis: Dict, fusion_analysis: Dict) -> Dict:
    """Classify enhanced forgery type using comprehensive analysis."""
    return enhanced_forgery_classifier.classify_forgery_type(
        visual_analysis, audio_analysis, temporal_analysis, fusion_analysis
    )
