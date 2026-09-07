"""
Root-level wrapper for report generator.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from report_generator import generate_pdf_report

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    out = generate_pdf_report(
        farmer_id="DEMO-FARMER-01",
        disease_name="Bacterial Blight",
        confidence_pct="99.2%",
        severity_level="Moderate Infection",
        severity_pct=28.5,
        treatment_info={
            "pesticides": "Copper Hydroxide 77% WP @ 2g/L",
            "organic": "Neem seed kernel extract 5%",
            "prevention": "Maintain field drainage",
            "recovery_time": "14 Days",
            "yield_impact": "15% loss"
        },
        output_path="reports/sample_report.pdf"
    )
    print("Report generator wrapper executed successfully:", out)
