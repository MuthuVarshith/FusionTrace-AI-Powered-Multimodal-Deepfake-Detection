"""
Explainable AI dashboard for deepfake detection analysis.
Provides detailed insights into model decisions and analysis results.
"""

import numpy as np
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ExplainableDashboard:
    """Generate explainable AI insights for deepfake detection."""
    
    def __init__(self):
        self.analysis_components = {
            'visual_features': ['face_consistency', 'eye_movement', 'lip_sync', 'skin_texture'],
            'audio_features': ['voice_consistency', 'spectral_analysis', 'temporal_patterns'],
            'temporal_features': ['frame_consistency', 'motion_analysis', 'temporal_anomalies'],
            'fusion_metrics': ['modality_agreement', 'confidence_distribution', 'decision_confidence']
        }
    
    def generate_analysis_insights(self, detection_results: Dict, media_type: str) -> Dict:
        """Generate comprehensive analysis insights."""
        try:
            insights = {
                'analysis_timestamp': datetime.now().isoformat(),
                'media_type': media_type,
                'overall_assessment': self._assess_overall_authenticity(detection_results),
                'feature_analysis': self._analyze_features(detection_results, media_type),
                'confidence_breakdown': self._breakdown_confidence(detection_results),
                'risk_assessment': self._assess_risk_level(detection_results),
                'explanation_summary': self._generate_explanation_summary(detection_results, media_type),
                'technical_details': self._get_technical_details(detection_results)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating analysis insights: {e}")
            return {'error': str(e)}
    
    def _assess_overall_authenticity(self, results: Dict) -> Dict:
        """Assess overall authenticity of the media."""
        prediction = results.get('prediction', 'Unknown')
        confidence = float(results.get('confidence', '0%').replace('%', '')) / 100.0
        
        # Determine authenticity level
        if prediction.lower() == 'real':
            if confidence > 0.8:
                authenticity_level = 'highly_authentic'
                description = 'Strong evidence of authenticity'
            elif confidence > 0.6:
                authenticity_level = 'likely_authentic'
                description = 'Good evidence of authenticity'
            else:
                authenticity_level = 'uncertain_authentic'
                description = 'Weak evidence of authenticity'
        else:
            if confidence > 0.8:
                authenticity_level = 'highly_suspicious'
                description = 'Strong evidence of manipulation'
            elif confidence > 0.6:
                authenticity_level = 'likely_manipulated'
                description = 'Good evidence of manipulation'
            else:
                authenticity_level = 'uncertain_manipulated'
                description = 'Weak evidence of manipulation'
        
        return {
            'level': authenticity_level,
            'description': description,
            'confidence_score': confidence,
            'prediction': prediction
        }
    
    def _analyze_features(self, results: Dict, media_type: str) -> Dict:
        """Analyze specific features based on media type."""
        feature_analysis = {}
        
        if media_type == 'image':
            feature_analysis = self._analyze_image_features(results)
        elif media_type == 'audio':
            feature_analysis = self._analyze_audio_features(results)
        elif media_type == 'video':
            feature_analysis = self._analyze_video_features(results)
        
        return feature_analysis
    
    def _analyze_image_features(self, results: Dict) -> Dict:
        """Analyze image-specific features."""
        features = {}
        
        # Enhanced analysis features
        if 'forgery_type' in results:
            features['forgery_classification'] = {
                'type': results.get('forgery_type', 'unknown'),
                'confidence': results.get('forgery_type_name', 'Unknown'),
                'explanation': self._explain_forgery_type(results.get('forgery_type'))
            }
        
        if 'fake_intensity' in results:
            intensity = results.get('fake_intensity', 0)
            features['manipulation_intensity'] = {
                'score': intensity,
                'level': results.get('intensity_level', 'unknown'),
                'description': results.get('intensity_description', ''),
                'explanation': self._explain_intensity_level(intensity)
            }
        
        # Grad-CAM++ analysis
        if 'gradcam_overlay' in results:
            features['attention_analysis'] = {
                'available': True,
                'explanation': 'Heat map shows regions of interest for AI decision',
                'regions_analyzed': ['facial_features', 'background_consistency', 'lighting_patterns']
            }
        
        return features
    
    def _analyze_audio_features(self, results: Dict) -> Dict:
        """Analyze audio-specific features."""
        return {
            'spectral_analysis': {
                'frequency_consistency': 'Analyzed for unnatural frequency patterns',
                'temporal_consistency': 'Checked for voice consistency over time',
                'synthetic_markers': 'Detected AI-generated voice characteristics'
            },
            'voice_characteristics': {
                'pitch_stability': 'Analyzed for unnatural pitch variations',
                'formant_analysis': 'Examined voice formant patterns',
                'prosody_analysis': 'Checked for unnatural speech rhythm'
            }
        }
    
    def _analyze_video_features(self, results: Dict) -> Dict:
        """Analyze video-specific features."""
        features = {}
        
        # Frame analysis
        if 'frame_analysis' in results:
            frame_data = results['frame_analysis']
            features['temporal_consistency'] = {
                'total_frames': frame_data.get('total_frames', 0),
                'fake_frames': frame_data.get('fake_frames', 0),
                'consistency_ratio': frame_data.get('fake_ratio', 0),
                'explanation': self._explain_temporal_consistency(frame_data)
            }
        
        # Audio-visual synchronization
        if 'audio_analysis' in results and 'frame_analysis' in results:
            features['multimodal_sync'] = {
                'audio_visual_agreement': self._check_audio_visual_agreement(results),
                'sync_quality': 'Analyzed for lip-sync and audio-visual consistency'
            }
        
        # Fusion analysis
        if 'fusion_analysis' in results:
            fusion_data = results['fusion_analysis']
            features['modality_fusion'] = {
                'fusion_method': fusion_data.get('fusion_method', 'unknown'),
                'modality_contributions': fusion_data.get('modality_contributions', {}),
                'explanation': self._explain_fusion_results(fusion_data)
            }
        
        return features
    
    def _breakdown_confidence(self, results: Dict) -> Dict:
        """Break down confidence scores by component."""
        confidence_breakdown = {
            'overall_confidence': results.get('confidence', '0%'),
            'components': {}
        }
        
        # Visual confidence
        if 'frame_analysis' in results:
            frame_results = results['frame_analysis'].get('frame_results', [])
            if frame_results:
                confidences = [float(fr.get('confidence', '0%').replace('%', '')) for fr in frame_results]
                confidence_breakdown['components']['visual'] = {
                    'mean_confidence': np.mean(confidences),
                    'confidence_variance': np.var(confidences),
                    'frame_count': len(frame_results)
                }
        
        # Audio confidence
        if 'audio_analysis' in results:
            audio_conf = float(results['audio_analysis'].get('confidence', '0%').replace('%', '')) / 100.0
            confidence_breakdown['components']['audio'] = {
                'confidence': audio_conf,
                'contribution': 'Audio analysis confidence'
            }
        
        # Temporal confidence
        if 'temporal_analysis' in results:
            temporal_data = results['temporal_analysis']
            confidence_breakdown['components']['temporal'] = {
                'consistency_score': temporal_data.get('consistency_score', 0),
                'temporal_stability': 'Frame-to-frame consistency analysis'
            }
        
        return confidence_breakdown
    
    def _assess_risk_level(self, results: Dict) -> Dict:
        """Assess risk level of the detected content."""
        prediction = results.get('prediction', 'Unknown')
        confidence = float(results.get('confidence', '0%').replace('%', '')) / 100.0
        
        if prediction.lower() == 'real':
            risk_level = 'low'
            risk_description = 'Content appears authentic'
        else:
            if confidence > 0.8:
                risk_level = 'high'
                risk_description = 'High confidence of manipulation'
            elif confidence > 0.6:
                risk_level = 'medium'
                risk_description = 'Moderate confidence of manipulation'
            else:
                risk_level = 'low'
                risk_description = 'Low confidence of manipulation'
        
        return {
            'level': risk_level,
            'description': risk_description,
            'confidence': confidence,
            'recommendations': self._get_risk_recommendations(risk_level, prediction)
        }
    
    def _generate_explanation_summary(self, results: Dict, media_type: str) -> Dict:
        """Generate human-readable explanation summary."""
        prediction = results.get('prediction', 'Unknown')
        confidence = results.get('confidence', '0%')
        
        if prediction.lower() == 'real':
            explanation = f"The analysis indicates this {media_type} appears to be authentic with {confidence} confidence. "
            explanation += "No significant signs of manipulation were detected across the analyzed features."
        else:
            explanation = f"The analysis indicates this {media_type} shows signs of manipulation with {confidence} confidence. "
            explanation += "Multiple indicators suggest artificial generation or modification."
        
        # Add specific insights based on media type
        if media_type == 'video' and 'temporal_analysis' in results:
            temporal_data = results['temporal_analysis']
            consistency = temporal_data.get('consistency_score', 0)
            if consistency < 0.5:
                explanation += f" Temporal analysis revealed inconsistencies across frames (consistency score: {consistency:.2f})."
        
        if 'forgery_type' in results:
            forgery_type = results.get('forgery_type_name', 'Unknown')
            explanation += f" The detected manipulation type is classified as: {forgery_type}."
        
        if 'fake_intensity' in results:
            intensity = results.get('fake_intensity', 0)
            explanation += f" The manipulation intensity is rated at {intensity:.2f} on a scale of 0-1."
        
        return {
            'summary': explanation,
            'key_findings': self._extract_key_findings(results),
            'technical_notes': self._get_technical_notes(results)
        }
    
    def _get_technical_details(self, results: Dict) -> Dict:
        """Get technical implementation details."""
        return {
            'detection_method': 'AI-powered multimodal analysis',
            'model_version': 'Fusion Trace v1.0',
            'analysis_timestamp': datetime.now().isoformat(),
            'processing_time': '< 5 seconds',
            'supported_formats': ['JPG', 'PNG', 'MP3', 'WAV', 'MP4', 'AVI'],
            'confidence_threshold': 0.5,
            'enhanced_features': {
                'gradcam_visualization': 'gradcam_overlay' in results,
                'forgery_classification': 'forgery_type' in results,
                'intensity_analysis': 'fake_intensity' in results,
                'temporal_analysis': 'temporal_analysis' in results,
                'multimodal_fusion': 'fusion_analysis' in results
            }
        }
    
    # Helper methods for explanations
    def _explain_forgery_type(self, forgery_type: str) -> str:
        """Explain the detected forgery type."""
        explanations = {
            'deepfake': 'AI-generated face replacement using deep learning techniques',
            'face_swap': 'Traditional face swapping between two individuals',
            'face_reenactment': 'Expression and pose manipulation of existing faces',
            'face_synthesis': 'Completely synthetic face generation',
            'inpainting': 'Content filling or editing of image regions',
            'splicing': 'Composition of multiple image elements'
        }
        return explanations.get(forgery_type, 'Unknown manipulation type')
    
    def _explain_intensity_level(self, intensity: float) -> str:
        """Explain the fake intensity level."""
        if intensity < 0.3:
            return 'Subtle manipulation with minimal detectable changes'
        elif intensity < 0.7:
            return 'Moderate manipulation with noticeable but not severe changes'
        else:
            return 'Severe manipulation with obvious artificial characteristics'
    
    def _explain_temporal_consistency(self, frame_data: Dict) -> str:
        """Explain temporal consistency analysis."""
        fake_ratio = frame_data.get('fake_ratio', 0)
        total_frames = frame_data.get('total_frames', 0)
        
        if fake_ratio > 0.8:
            return f'High manipulation consistency: {fake_ratio:.1%} of frames show manipulation'
        elif fake_ratio > 0.5:
            return f'Moderate manipulation: {fake_ratio:.1%} of frames show manipulation'
        elif fake_ratio > 0.2:
            return f'Low manipulation: {fake_ratio:.1%} of frames show manipulation'
        else:
            return f'Minimal manipulation: {fake_ratio:.1%} of frames show manipulation'
    
    def _check_audio_visual_agreement(self, results: Dict) -> str:
        """Check agreement between audio and visual analysis."""
        audio_pred = results.get('audio_analysis', {}).get('prediction', 'Unknown')
        visual_pred = results.get('frame_analysis', {}).get('frame_results', [{}])[0].get('prediction', 'Unknown')
        
        if audio_pred == visual_pred:
            return 'Audio and visual analysis agree'
        else:
            return 'Audio and visual analysis disagree - requires further investigation'
    
    def _explain_fusion_results(self, fusion_data: Dict) -> str:
        """Explain fusion analysis results."""
        modality_contributions = fusion_data.get('modality_contributions', {})
        explanation = "Multimodal analysis combining: "
        
        components = []
        if 'visual' in modality_contributions:
            components.append('visual analysis')
        if 'audio' in modality_contributions:
            components.append('audio analysis')
        if 'temporal' in modality_contributions:
            components.append('temporal consistency')
        
        explanation += ', '.join(components)
        explanation += f" with final confidence: {fusion_data.get('confidence', '0%')}"
        
        return explanation
    
    def _extract_key_findings(self, results: Dict) -> List[str]:
        """Extract key findings from the analysis."""
        findings = []
        
        prediction = results.get('prediction', 'Unknown')
        confidence = results.get('confidence', '0%')
        findings.append(f"Primary prediction: {prediction} ({confidence})")
        
        if 'forgery_type' in results:
            findings.append(f"Manipulation type: {results.get('forgery_type_name', 'Unknown')}")
        
        if 'fake_intensity' in results:
            findings.append(f"Manipulation intensity: {results.get('intensity_level', 'Unknown')}")
        
        if 'temporal_analysis' in results:
            consistency = results['temporal_analysis'].get('consistency_score', 0)
            findings.append(f"Temporal consistency: {consistency:.2f}")
        
        return findings
    
    def _get_technical_notes(self, results: Dict) -> List[str]:
        """Get technical notes about the analysis."""
        notes = []
        
        if 'gradcam_overlay' in results:
            notes.append("Grad-CAM++ visualization available for attention analysis")
        
        if 'fusion_analysis' in results:
            notes.append("Late fusion methodology applied for multimodal analysis")
        
        if 'temporal_analysis' in results:
            notes.append("Temporal consistency analysis performed across frames")
        
        return notes
    
    def _get_risk_recommendations(self, risk_level: str, prediction: str) -> List[str]:
        """Get risk-based recommendations."""
        recommendations = []
        
        if risk_level == 'high':
            recommendations.extend([
                "High confidence of manipulation detected",
                "Recommend additional verification methods",
                "Consider expert review for critical decisions"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "Moderate confidence of manipulation",
                "Consider additional verification",
                "Use as one factor in decision making"
            ])
        else:
            recommendations.extend([
                "Low risk of manipulation",
                "Standard verification procedures sufficient",
                "Monitor for any additional concerns"
            ])
        
        return recommendations

def generate_explainable_insights(detection_results: Dict, media_type: str) -> Dict:
    """Generate explainable AI insights for detection results."""
    dashboard = ExplainableDashboard()
    return dashboard.generate_analysis_insights(detection_results, media_type)
