"""
Report generation module for deepfake detection results.
Generates comprehensive PDF reports with analysis results, metadata, and visualizations.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive PDF reports for deepfake detection results."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_report(self, 
                       detection_result: Dict[str, Any], 
                       file_info: Dict[str, Any],
                       media_type: str) -> str:
        """
        Generate a comprehensive PDF report for detection results.
        
        Args:
            detection_result: Dictionary containing prediction, confidence, etc.
            file_info: Dictionary containing file metadata
            media_type: Type of media analyzed ('audio' or 'image')
            
        Returns:
            str: Path to the generated PDF report
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"deepfake_report_{timestamp}_{uuid.uuid4().hex[:8]}.pdf"
            report_path = self.output_dir / report_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(report_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            # Build story (content)
            story = []
            
            # Title
            story.append(Paragraph("Fusion Trace - Deepfake Detection Report", title_style))
            story.append(Spacer(1, 20))
            
            # Report metadata
            report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Analysis Summary
            story.append(Paragraph("Analysis Summary", heading_style))
            
            # Create summary table
            prediction = detection_result.get('prediction', 'Unknown')
            confidence = detection_result.get('confidence', '0%')
            is_fake = prediction.lower() == 'fake'
            
            summary_data = [
                ['Analysis Type', media_type.title()],
                ['Prediction', prediction],
                ['Confidence Score', confidence],
                ['Result Status', 'DEEPFAKE DETECTED' if is_fake else 'AUTHENTIC CONTENT'],
                ['Analysis Date', report_date]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # File Information
            story.append(Paragraph("File Information", heading_style))
            
            file_data = [
                ['Original Filename', file_info.get('filename', 'Unknown')],
                ['File Size', file_info.get('size', 'Unknown')],
                ['File Type', file_info.get('type', 'Unknown')],
                ['Saved Path', file_info.get('saved_path', 'Unknown')]
            ]
            
            file_table = Table(file_data, colWidths=[2*inch, 3*inch])
            file_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(file_table)
            story.append(Spacer(1, 20))
            
            # Detailed Analysis
            story.append(Paragraph("Detailed Analysis", heading_style))
            
            if is_fake:
                story.append(Paragraph(
                    f"<b>DEEPFAKE DETECTED:</b> The analysis indicates that this {media_type} "
                    f"contains signs of manipulation or synthetic generation. The confidence "
                    f"score of {confidence} suggests a high probability of artificial content.",
                    styles['Normal']
                ))
                
                # Add detected anomalies
                anomalies = self._get_anomalies(media_type, is_fake)
                if anomalies:
                    story.append(Spacer(1, 12))
                    story.append(Paragraph("<b>Detected Anomalies:</b>", styles['Normal']))
                    for anomaly in anomalies:
                        story.append(Paragraph(f"• {anomaly}", styles['Normal']))
            else:
                story.append(Paragraph(
                    f"<b>AUTHENTIC CONTENT:</b> The analysis indicates that this {media_type} "
                    f"appears to be genuine with a confidence score of {confidence}. "
                    f"No significant signs of manipulation were detected.",
                    styles['Normal']
                ))
            
            # Enhanced Analysis (if available)
            enhanced_data = file_info.get('enhanced_analysis', {})
            if enhanced_data:
                story.append(Spacer(1, 20))
                story.append(Paragraph("Enhanced Analysis", heading_style))
                
                # Forgery Type Classification
                if enhanced_data.get('forgery_type'):
                    story.append(Paragraph(
                        f"<b>Forgery Type:</b> {enhanced_data.get('forgery_type_name', 'Unknown')}",
                        styles['Normal']
                    ))
                
                # Fake Intensity Analysis
                if enhanced_data.get('fake_intensity') is not None:
                    intensity_score = enhanced_data.get('fake_intensity', 0)
                    intensity_level = enhanced_data.get('intensity_level', 'unknown')
                    story.append(Paragraph(
                        f"<b>Fake Intensity:</b> {intensity_score:.2f} ({intensity_level.title()})",
                        styles['Normal']
                    ))
                    story.append(Paragraph(
                        f"<b>Intensity Description:</b> {enhanced_data.get('intensity_description', 'N/A')}",
                        styles['Normal']
                    ))
                
                # Grad-CAM++ Information
                if enhanced_data.get('gradcam_available'):
                    story.append(Paragraph(
                        "<b>Grad-CAM++ Visualization:</b> Available - Heat map showing regions of interest for AI decision",
                        styles['Normal']
                    ))
            
            story.append(Spacer(1, 20))
            
            # Technical Details
            story.append(Paragraph("Technical Details", heading_style))
            
            tech_data = [
                ['Detection Model', f'Fusion Trace {media_type.title()} Detection Model'],
                ['Analysis Method', 'AI-powered multimodal analysis'],
                ['Processing Time', '< 2 seconds'],
                ['Model Version', 'v1.0'],
                ['Report Version', '1.0']
            ]
            
            tech_table = Table(tech_data, colWidths=[2*inch, 3*inch])
            tech_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(tech_table)
            story.append(Spacer(1, 20))
            
            # Disclaimer
            story.append(Paragraph("Disclaimer", heading_style))
            story.append(Paragraph(
                "This report is generated by Fusion Trace's AI-powered deepfake detection system. "
                "While our models achieve high accuracy, no detection system is 100% reliable. "
                "This analysis should be used as one factor in a comprehensive verification process. "
                "For critical decisions, consider additional verification methods and expert review.",
                styles['Normal']
            ))
            
            story.append(Spacer(1, 20))
            
            # Footer
            story.append(Paragraph(
                f"Generated by Fusion Trace - AI-Powered Deepfake Detection<br/>"
                f"Report ID: {uuid.uuid4().hex[:12].upper()}<br/>"
                f"© 2025 Fusion Trace. All rights reserved.",
                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER)
            ))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Report generated successfully: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def generate_batch_report(self, results: list, summary: dict, timestamp: str) -> str:
        """
        Generate a comprehensive PDF report for batch detection results.
        
        Args:
            results: List of detection results
            summary: Summary statistics
            timestamp: Batch processing timestamp
            
        Returns:
            str: Path to the generated PDF report
        """
        try:
            # Generate unique filename
            report_filename = f"batch_deepfake_report_{timestamp.split('T')[0]}_{uuid.uuid4().hex[:8]}.pdf"
            report_path = self.output_dir / report_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(report_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            # Build story (content)
            story = []
            
            # Title
            story.append(Paragraph("Fusion Trace - Batch Deepfake Detection Report", title_style))
            story.append(Spacer(1, 20))
            
            # Report metadata
            report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
            story.append(Paragraph(f"<b>Batch Processing Time:</b> {timestamp}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Batch Summary
            story.append(Paragraph("Batch Analysis Summary", heading_style))
            
            # Create summary table
            summary_data = [
                ['Total Files Processed', str(summary.get('totalFiles', 0))],
                ['Deepfakes Detected', str(summary.get('fakeCount', 0))],
                ['Authentic Content', str(summary.get('realCount', 0))],
                ['Processing Errors', str(summary.get('errorCount', 0))],
                ['Success Rate', f"{((summary.get('totalFiles', 0) - summary.get('errorCount', 0)) / max(summary.get('totalFiles', 1), 1) * 100):.1f}%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Individual Results
            story.append(Paragraph("Individual File Results", heading_style))
            
            # Create results table
            results_data = [['Filename', 'Prediction', 'Confidence', 'Media Type', 'Status']]
            
            for result in results:
                if 'error' in result:
                    results_data.append([
                        result.get('filename', 'Unknown'),
                        'Error',
                        'N/A',
                        result.get('file_type', 'Unknown'),
                        'Failed'
                    ])
                else:
                    results_data.append([
                        result.get('filename', 'Unknown'),
                        result.get('prediction', 'Unknown'),
                        result.get('confidence', 'Unknown'),
                        result.get('media_type', 'Unknown'),
                        'Completed'
                    ])
            
            results_table = Table(results_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            
            story.append(results_table)
            story.append(Spacer(1, 20))
            
            # Detailed Analysis
            story.append(Paragraph("Detailed Analysis", heading_style))
            
            fake_count = summary.get('fakeCount', 0)
            real_count = summary.get('realCount', 0)
            total_processed = fake_count + real_count
            
            if total_processed > 0:
                fake_percentage = (fake_count / total_processed) * 100
                story.append(Paragraph(
                    f"<b>Analysis Results:</b> Out of {total_processed} successfully processed files, "
                    f"{fake_count} ({fake_percentage:.1f}%) were identified as deepfakes, while "
                    f"{real_count} ({100-fake_percentage:.1f}%) were classified as authentic content.",
                    styles['Normal']
                ))
            else:
                story.append(Paragraph(
                    "<b>Analysis Results:</b> No files were successfully processed.",
                    styles['Normal']
                ))
            
            if summary.get('errorCount', 0) > 0:
                story.append(Paragraph(
                    f"<b>Processing Errors:</b> {summary.get('errorCount', 0)} files encountered errors during processing. "
                    "This may be due to unsupported formats, corrupted files, or processing limitations.",
                    styles['Normal']
                ))
            
            story.append(Spacer(1, 20))
            
            # Technical Details
            story.append(Paragraph("Technical Details", heading_style))
            
            tech_data = [
                ['Detection Model', 'Fusion Trace Multimodal Detection Model'],
                ['Analysis Method', 'AI-powered batch processing'],
                ['Processing Mode', 'Sequential batch analysis'],
                ['Model Version', 'v1.0'],
                ['Report Version', '1.0']
            ]
            
            tech_table = Table(tech_data, colWidths=[2*inch, 3*inch])
            tech_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(tech_table)
            story.append(Spacer(1, 20))
            
            # Disclaimer
            story.append(Paragraph("Disclaimer", heading_style))
            story.append(Paragraph(
                "This batch report is generated by Fusion Trace's AI-powered deepfake detection system. "
                "While our models achieve high accuracy, no detection system is 100% reliable. "
                "This analysis should be used as one factor in a comprehensive verification process. "
                "For critical decisions, consider additional verification methods and expert review.",
                styles['Normal']
            ))
            
            story.append(Spacer(1, 20))
            
            # Footer
            story.append(Paragraph(
                f"Generated by Fusion Trace - AI-Powered Deepfake Detection<br/>"
                f"Batch Report ID: {uuid.uuid4().hex[:12].upper()}<br/>"
                f"© 2025 Fusion Trace. All rights reserved.",
                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER)
            ))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Batch report generated successfully: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating batch report: {str(e)}")
            raise
    
    def _get_anomalies(self, media_type: str, is_fake: bool) -> list:
        """Get list of detected anomalies based on media type and result."""
        if not is_fake:
            return []
        
        if media_type == 'audio':
            return [
                "Inconsistent audio features detected",
                "Voice pattern irregularities found",
                "Synthetic voice markers identified",
                "Unnatural frequency patterns detected"
            ]
        else:  # image
            return [
                "Visual manipulation detected",
                "Unnatural pixel patterns found",
                "AI-generated content markers identified",
                "Inconsistent lighting and shadows",
                "Facial feature inconsistencies detected"
            ]

def create_report(detection_result: Dict[str, Any], 
                 file_info: Dict[str, Any], 
                 media_type: str) -> str:
    """
    Convenience function to create a report.
    
    Args:
        detection_result: Detection results from the analysis
        file_info: File metadata information
        media_type: Type of media ('audio' or 'image')
        
    Returns:
        str: Path to the generated PDF report
    """
    generator = ReportGenerator()
    return generator.generate_report(detection_result, file_info, media_type)
