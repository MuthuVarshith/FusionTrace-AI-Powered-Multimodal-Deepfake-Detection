"""
Advanced visualization engine for deepfake detection.
Includes Grad-CAM++ heatmaps, confidence timelines, side-by-side visualization, and confidence gauges.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image, ImageDraw, ImageFont
import json

logger = logging.getLogger(__name__)

class GradCAMVisualizer:
    """Enhanced Grad-CAM++ visualization with region highlighting."""
    
    def __init__(self):
        self.colormap = plt.cm.jet
        self.attention_colors = {
            'high': (255, 0, 0),      # Red for high attention
            'medium': (255, 165, 0),  # Orange for medium attention
            'low': (0, 255, 0)        # Green for low attention
        }
    
    def create_enhanced_heatmap(self, original_image: np.ndarray, 
                               gradcam_heatmap: np.ndarray,
                               attention_regions: List[Dict] = None) -> Dict:
        """Create enhanced Grad-CAM++ visualization with region highlighting."""
        try:
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(gradcam_heatmap, 
                                       (original_image.shape[1], original_image.shape[0]))
            
            # Create attention overlay
            attention_overlay = self._create_attention_overlay(
                original_image, heatmap_resized, attention_regions
            )
            
            # Create side-by-side visualization
            side_by_side = self._create_side_by_side_viz(
                original_image, heatmap_resized, attention_overlay
            )
            
            # Generate region analysis
            region_analysis = self._analyze_attention_regions(
                gradcam_heatmap, attention_regions
            )
            
            return {
                'heatmap_overlay': self._array_to_base64(attention_overlay),
                'side_by_side': self._array_to_base64(side_by_side),
                'region_analysis': region_analysis,
                'attention_summary': self._generate_attention_summary(region_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error creating enhanced heatmap: {e}")
            return {'error': str(e)}
    
    def _create_attention_overlay(self, original_image: np.ndarray, 
                               heatmap: np.ndarray, 
                               attention_regions: List[Dict] = None) -> np.ndarray:
        """Create attention overlay with region highlighting."""
        # Normalize heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Apply colormap
        heatmap_colored = self.colormap(heatmap_norm)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Blend with original image
        overlay = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)
        
        # Highlight specific regions if provided
        if attention_regions:
            overlay = self._highlight_regions(overlay, attention_regions)
        
        return overlay
    
    def _highlight_regions(self, image: np.ndarray, 
                          regions: List[Dict]) -> np.ndarray:
        """Highlight specific attention regions."""
        highlighted = image.copy()
        
        for region in regions:
            x, y, w, h = region.get('bbox', [0, 0, 0, 0])
            attention_level = region.get('attention_level', 'medium')
            color = self.attention_colors.get(attention_level, (255, 165, 0))
            
            # Draw bounding box
            cv2.rectangle(highlighted, (x, y), (x + w, y + h), color, 3)
            
            # Add label
            label = f"{region.get('region_type', 'Region')}: {attention_level}"
            cv2.putText(highlighted, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return highlighted
    
    def _create_side_by_side_viz(self, original: np.ndarray, 
                                heatmap: np.ndarray, 
                                overlay: np.ndarray) -> np.ndarray:
        """Create side-by-side visualization."""
        # Resize images to same height
        target_height = 300
        original_resized = cv2.resize(original, 
                                     (int(original.shape[1] * target_height / original.shape[0]), 
                                      target_height))
        heatmap_resized = cv2.resize(heatmap, 
                                    (int(heatmap.shape[1] * target_height / heatmap.shape[0]), 
                                     target_height))
        overlay_resized = cv2.resize(overlay, 
                                   (int(overlay.shape[1] * target_height / overlay.shape[0]), 
                                    target_height))
        
        # Create side-by-side layout
        side_by_side = np.hstack([original_resized, heatmap_resized, overlay_resized])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Heatmap", (original_resized.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Overlay", (original_resized.shape[1] + heatmap_resized.shape[1] + 10, 30), 
                   font, 1, (255, 255, 255), 2)
        
        return side_by_side
    
    def _analyze_attention_regions(self, heatmap: np.ndarray, 
                                 regions: List[Dict] = None) -> Dict:
        """Analyze attention regions and their characteristics."""
        # Find high attention areas
        high_attention_threshold = 0.8
        high_attention_mask = heatmap > high_attention_threshold
        
        # Calculate attention statistics
        attention_stats = {
            'max_attention': float(np.max(heatmap)),
            'mean_attention': float(np.mean(heatmap)),
            'high_attention_ratio': float(np.sum(high_attention_mask) / heatmap.size),
            'attention_variance': float(np.var(heatmap))
        }
        
        # Analyze specific regions
        region_analysis = []
        if regions:
            for region in regions:
                bbox = region.get('bbox', [0, 0, 0, 0])
                x, y, w, h = bbox
                if x + w <= heatmap.shape[1] and y + h <= heatmap.shape[0]:
                    region_heatmap = heatmap[y:y+h, x:x+w]
                    region_analysis.append({
                        'region_type': region.get('region_type', 'unknown'),
                        'attention_score': float(np.mean(region_heatmap)),
                        'attention_variance': float(np.var(region_heatmap)),
                        'bbox': bbox
                    })
        
        return {
            'attention_stats': attention_stats,
            'region_analysis': region_analysis,
            'heatmap_shape': heatmap.shape
        }
    
    def _generate_attention_summary(self, region_analysis: Dict) -> str:
        """Generate human-readable attention summary."""
        stats = region_analysis.get('attention_stats', {})
        max_attention = stats.get('max_attention', 0)
        high_ratio = stats.get('high_attention_ratio', 0)
        
        if max_attention > 0.8:
            if high_ratio > 0.3:
                return "High attention concentration detected - strong manipulation indicators"
            else:
                return "Focused attention areas detected - specific manipulation regions"
        elif max_attention > 0.5:
            return "Moderate attention distribution - some manipulation indicators"
        else:
            return "Low attention concentration - minimal manipulation indicators"
    
    def _array_to_base64(self, image_array: np.ndarray) -> str:
        """Convert numpy array to base64 string."""
        try:
            # Convert to PIL Image
            if len(image_array.shape) == 3:
                pil_image = Image.fromarray(image_array)
            else:
                pil_image = Image.fromarray((image_array * 255).astype(np.uint8))
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"Error converting array to base64: {e}")
            return ""

class ConfidenceTimeline:
    """Create confidence timeline for audio analysis."""
    
    def __init__(self):
        self.timeline_colors = {
            'high_confidence': '#10b981',
            'medium_confidence': '#f59e0b',
            'low_confidence': '#ef4444'
        }
    
    def create_audio_timeline(self, audio_confidence_data: List[Dict]) -> Dict:
        """Create confidence timeline for audio analysis."""
        try:
            # Extract timeline data
            timestamps = [point['timestamp'] for point in audio_confidence_data]
            confidences = [point['confidence'] for point in audio_confidence_data]
            predictions = [point['prediction'] for point in audio_confidence_data]
            
            # Create timeline plot
            fig = go.Figure()
            
            # Add confidence line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=6)
            ))
            
            # Add prediction regions
            self._add_prediction_regions(fig, timestamps, predictions, confidences)
            
            # Add manipulation markers
            manipulation_points = self._find_manipulation_points(confidences, timestamps)
            if manipulation_points:
                self._add_manipulation_markers(fig, manipulation_points)
            
            # Update layout
            fig.update_layout(
                title='Audio Confidence Timeline',
                xaxis_title='Time (seconds)',
                yaxis_title='Confidence Score',
                yaxis=dict(range=[0, 1]),
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Convert to HTML
            timeline_html = fig.to_html(include_plotlyjs=False, div_id="confidence-timeline")
            
            return {
                'timeline_html': timeline_html,
                'manipulation_points': manipulation_points,
                'confidence_stats': self._calculate_confidence_stats(confidences),
                'timeline_summary': self._generate_timeline_summary(manipulation_points, confidences)
            }
            
        except Exception as e:
            logger.error(f"Error creating confidence timeline: {e}")
            return {'error': str(e)}
    
    def _add_prediction_regions(self, fig, timestamps, predictions, confidences):
        """Add prediction regions to timeline."""
        current_prediction = None
        start_time = None
        
        for i, (timestamp, prediction) in enumerate(zip(timestamps, predictions)):
            if prediction != current_prediction:
                if current_prediction is not None:
                    # Add region
                    fig.add_vrect(
                        x0=start_time, x1=timestamp,
                        fillcolor=self._get_prediction_color(current_prediction),
                        opacity=0.3,
                        annotation_text=current_prediction,
                        annotation_position="top"
                    )
                
                current_prediction = prediction
                start_time = timestamp
        
        # Add final region
        if current_prediction is not None:
            fig.add_vrect(
                x0=start_time, x1=timestamps[-1],
                fillcolor=self._get_prediction_color(current_prediction),
                opacity=0.3,
                annotation_text=current_prediction,
                annotation_position="top"
            )
    
    def _get_prediction_color(self, prediction):
        """Get color for prediction type."""
        return '#ef4444' if prediction.lower() == 'fake' else '#10b981'
    
    def _find_manipulation_points(self, confidences, timestamps):
        """Find points where manipulation starts/ends."""
        manipulation_points = []
        threshold = 0.7  # Confidence threshold for manipulation
        
        for i, confidence in enumerate(confidences):
            if confidence > threshold:
                # Check if this is a start or end of manipulation
                prev_conf = confidences[i-1] if i > 0 else 0
                next_conf = confidences[i+1] if i < len(confidences)-1 else 0
                
                if prev_conf <= threshold and confidence > threshold:
                    manipulation_points.append({
                        'timestamp': timestamps[i],
                        'type': 'manipulation_start',
                        'confidence': confidence
                    })
                elif confidence > threshold and next_conf <= threshold:
                    manipulation_points.append({
                        'timestamp': timestamps[i],
                        'type': 'manipulation_end',
                        'confidence': confidence
                    })
        
        return manipulation_points
    
    def _add_manipulation_markers(self, fig, manipulation_points):
        """Add manipulation markers to timeline."""
        for point in manipulation_points:
            fig.add_vline(
                x=point['timestamp'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"{point['type']}: {point['confidence']:.2f}",
                annotation_position="top"
            )
    
    def _calculate_confidence_stats(self, confidences):
        """Calculate confidence statistics."""
        return {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'max_confidence': float(np.max(confidences)),
            'min_confidence': float(np.min(confidences)),
            'high_confidence_ratio': float(np.sum(np.array(confidences) > 0.7) / len(confidences))
        }
    
    def _generate_timeline_summary(self, manipulation_points, confidences):
        """Generate timeline summary."""
        if not manipulation_points:
            return "No significant manipulation detected in audio timeline"
        
        start_points = [p for p in manipulation_points if p['type'] == 'manipulation_start']
        end_points = [p for p in manipulation_points if p['type'] == 'manipulation_end']
        
        return f"Detected {len(start_points)} manipulation periods with confidence peaks at {len(manipulation_points)} points"

class ConfidenceGauge:
    """Create confidence gauge visualization."""
    
    def __init__(self):
        self.gauge_colors = {
            'high': '#10b981',
            'medium': '#f59e0b',
            'low': '#ef4444'
        }
    
    def create_confidence_gauge(self, confidence: float, prediction: str, 
                              modality_breakdown: Dict = None) -> Dict:
        """Create confidence gauge visualization."""
        try:
            # Create gauge figure
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Confidence: {prediction}"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                title="Deepfake Detection Confidence",
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            # Convert to HTML
            gauge_html = fig.to_html(include_plotlyjs=False, div_id="confidence-gauge")
            
            # Create modality breakdown if provided
            breakdown_html = ""
            if modality_breakdown:
                breakdown_html = self._create_modality_breakdown(modality_breakdown)
            
            return {
                'gauge_html': gauge_html,
                'breakdown_html': breakdown_html,
                'confidence_level': self._get_confidence_level(confidence),
                'gauge_summary': self._generate_gauge_summary(confidence, prediction)
            }
            
        except Exception as e:
            logger.error(f"Error creating confidence gauge: {e}")
            return {'error': str(e)}
    
    def _create_modality_breakdown(self, modality_breakdown: Dict) -> str:
        """Create modality breakdown visualization."""
        modalities = list(modality_breakdown.keys())
        confidences = [modality_breakdown[mod]['confidence'] for mod in modalities]
        
        fig = go.Figure(data=[
            go.Bar(
                x=modalities,
                y=confidences,
                marker_color=[self._get_confidence_color(conf) for conf in confidences]
            )
        ])
        
        fig.update_layout(
            title="Modality Confidence Breakdown",
            xaxis_title="Modality",
            yaxis_title="Confidence",
            yaxis=dict(range=[0, 1])
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="modality-breakdown")
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level string."""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color for confidence level."""
        level = self._get_confidence_level(confidence)
        return self.gauge_colors[level]
    
    def _generate_gauge_summary(self, confidence: float, prediction: str) -> str:
        """Generate gauge summary."""
        level = self._get_confidence_level(confidence)
        return f"{prediction} prediction with {level} confidence ({confidence:.1%})"

class VisualizationEngine:
    """Main visualization engine combining all visualization components."""
    
    def __init__(self):
        self.gradcam_visualizer = GradCAMVisualizer()
        self.timeline_creator = ConfidenceTimeline()
        self.gauge_creator = ConfidenceGauge()
    
    def create_comprehensive_visualization(self, analysis_results: Dict) -> Dict:
        """Create comprehensive visualization package."""
        try:
            visualizations = {}
            
            # Grad-CAM++ visualization
            if 'gradcam_heatmap' in analysis_results:
                gradcam_result = self.gradcam_visualizer.create_enhanced_heatmap(
                    analysis_results.get('original_image'),
                    analysis_results.get('gradcam_heatmap'),
                    analysis_results.get('attention_regions', [])
                )
                visualizations['gradcam'] = gradcam_result
            
            # Confidence timeline
            if 'audio_timeline' in analysis_results:
                timeline_result = self.timeline_creator.create_audio_timeline(
                    analysis_results.get('audio_timeline')
                )
                visualizations['timeline'] = timeline_result
            
            # Confidence gauge
            confidence = float(analysis_results.get('confidence', '0%').replace('%', '')) / 100.0
            prediction = analysis_results.get('prediction', 'Unknown')
            modality_breakdown = analysis_results.get('modality_breakdown', {})
            
            gauge_result = self.gauge_creator.create_confidence_gauge(
                confidence, prediction, modality_breakdown
            )
            visualizations['gauge'] = gauge_result
            
            return {
                'visualizations': visualizations,
                'summary': self._generate_visualization_summary(visualizations),
                'interactive_elements': self._get_interactive_elements(visualizations)
            }
            
        except Exception as e:
            logger.error(f"Error creating comprehensive visualization: {e}")
            return {'error': str(e)}
    
    def _generate_visualization_summary(self, visualizations: Dict) -> str:
        """Generate visualization summary."""
        available_viz = list(visualizations.keys())
        return f"Generated {len(available_viz)} visualization components: {', '.join(available_viz)}"
    
    def _get_interactive_elements(self, visualizations: Dict) -> List[str]:
        """Get list of interactive elements."""
        interactive = []
        if 'gradcam' in visualizations:
            interactive.append('Grad-CAM++ heatmap overlay')
        if 'timeline' in visualizations:
            interactive.append('Audio confidence timeline')
        if 'gauge' in visualizations:
            interactive.append('Confidence gauge')
        return interactive

# Global instance
visualization_engine = VisualizationEngine()

def create_enhanced_visualizations(analysis_results: Dict) -> Dict:
    """Create enhanced visualizations for analysis results."""
    return visualization_engine.create_comprehensive_visualization(analysis_results)
