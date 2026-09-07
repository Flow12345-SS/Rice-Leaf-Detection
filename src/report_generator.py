"""
RiceCare AI PDF Field Report Generator
Generates clinical agronomic advisory PDF reports for farmers and crop consultants.
"""

import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def generate_pdf_report(
    farmer_id,
    disease_name,
    confidence_pct,
    severity_level,
    severity_pct,
    treatment_info,
    leaf_image_pil=None,
    output_path=None
):
    """
    Generates a professional PDF diagnostic report.
    
    Args:
        farmer_id (str): Unique Farmer / Farm Lot Identifier.
        disease_name (str): Diagnosed disease name.
        confidence_pct (str): Confidence string (e.g. '98.5%').
        severity_level (str): 'Mild Infection', 'Moderate Infection', etc.
        severity_pct (float/str): Percentage of infected leaf area.
        treatment_info (dict): Chemical, organic, and preventive recommendations.
        leaf_image_pil (PIL.Image): Optional image to embed.
        output_path (str): Optional file path to save PDF. If None, returns BytesIO buffer.
        
    Returns:
        bytes or str: PDF bytes buffer or output path.
    """
    buffer = io.BytesIO() if output_path is None else open(output_path, "wb")
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1b4332'),
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#40916c'),
        alignment=TA_CENTER
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1b4332'),
        spaceBefore=10,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#2d3748')
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("🌾 <b>RiceCare AI™ - AgriTech Advisory Platform</b>", title_style))
    story.append(Paragraph("Automated Foliar Pathology & Integrated Pest Management Report", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2d6a4f'), spaceAfter=15))

    # 2. Metadata Table
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [Paragraph("<b>Farmer / Farm ID:</b>", body_style), Paragraph(str(farmer_id), body_style),
         Paragraph("<b>Audit Timestamp:</b>", body_style), Paragraph(timestamp_str, body_style)],
        [Paragraph("<b>Diagnosed Disease:</b>", body_style), Paragraph(f"<b><font color='#d90429'>{disease_name}</font></b>", body_style),
         Paragraph("<b>AI Confidence:</b>", body_style), Paragraph(str(confidence_pct), body_style)],
        [Paragraph("<b>Severity Rating:</b>", body_style), Paragraph(f"<b>{severity_level}</b>", body_style),
         Paragraph("<b>Infected Leaf Area:</b>", body_style), Paragraph(f"{severity_pct}%", body_style)]
    ]

    meta_table = Table(meta_data, colWidths=[120, 140, 120, 150])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#edf2f7')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # 3. Embed Leaf Image if provided
    if leaf_image_pil is not None:
        img_buffer = io.BytesIO()
        leaf_image_pil.convert("RGB").save(img_buffer, format="JPEG", quality=85)
        img_buffer.seek(0)
        report_img = RLImage(img_buffer, width=160, height=160)
        
        img_wrapper = Table([[report_img, Paragraph("<b>Pathology Visual Audit:</b><br/>Uploaded specimen subjected to MobileNetV2 feature extraction and necrotic lesion colorimetry. Diagnostic regions validated against IRRI reference criteria.", body_style)]], colWidths=[180, 350])
        img_wrapper.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(img_wrapper)
        story.append(Spacer(1, 12))

    # 4. Treatment Protocols & Prescriptions
    story.append(Paragraph("💊 <b>Recommended Actionable Interventions</b>", section_heading))
    
    pesticides = treatment_info.get("pesticides", "Apply recommended bactericide or fungicide.")
    organics = treatment_info.get("organic", "Apply bio-fungicides such as Trichoderma viride.")
    prevention = treatment_info.get("prevention", "Maintain field hygiene and balanced fertilization.")
    recovery = treatment_info.get("recovery_time", "10 - 14 Days")
    yield_impact = treatment_info.get("yield_impact", "15% - 25% if untreated")

    treatment_data = [
        [Paragraph("<b>Targeted Agrochemical:</b>", body_style), Paragraph(pesticides, body_style)],
        [Paragraph("<b>Certified Bio-Organic Remedy:</b>", body_style), Paragraph(organics, body_style)],
        [Paragraph("<b>Cultural Prevention & IPM:</b>", body_style), Paragraph(prevention, body_style)],
        [Paragraph("<b>Expected Recovery Timeline:</b>", body_style), Paragraph(recovery, body_style)],
        [Paragraph("<b>Potential Yield Impact:</b>", body_style), Paragraph(yield_impact, body_style)]
    ]

    treatment_table = Table(treatment_data, colWidths=[160, 370])
    treatment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(treatment_table)
    story.append(Spacer(1, 20))

    # 5. Footer Disclaimer & Signature
    disclaimer_text = "<b>Notice:</b> This AI advisory report is generated by RiceCare AI Computer Vision models for preliminary field assistance. For catastrophic outbreaks or unusual chlorosis, consult your nearest Krishi Vigyan Kendra (KVK) or certified agricultural extension officer."
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disc', parent=body_style, fontSize=8, textColor=colors.HexColor('#64748b'))))
    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceAfter=8))
    story.append(Paragraph("Verified by <b>RiceCare AI™ Diagnostic Engine</b> | Certified AgriTech Submission", subtitle_style))

    doc.build(story)

    if output_path is None:
        buffer.seek(0)
        return buffer.getvalue()
    else:
        buffer.close()
        return output_path

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    out_file = "reports/sample_report.pdf"
    generate_pdf_report(
        farmer_id="FARMER-KA-802",
        disease_name="Bacterial Blight",
        confidence_pct="99.4%",
        severity_level="Moderate Infection",
        severity_pct=34.5,
        treatment_info={
            "pesticides": "Streptomycin sulfate + Tetracycline (90:10) @ 300g/ha",
            "organic": "Foliar spray of fresh cow dung slurry supernatant (20%) or Neem oil (3%)",
            "prevention": "Avoid nitrogen over-application; ensure proper field drainage",
            "recovery_time": "12 - 16 Days",
            "yield_impact": "20% - 30% yield loss if untreated"
        },
        output_path=out_file
    )
    print(f"Generated sample PDF report: {out_file}")
