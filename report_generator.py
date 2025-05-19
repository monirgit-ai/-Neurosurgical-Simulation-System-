from fpdf import FPDF
from datetime import datetime
import os
import cv2

class SimulationReport(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, "Theseus Simulation Report", ln=True, align='C')
        self.set_font("Arial", '', 10)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        self.ln(10)

    def add_simulation_data(self, data):
        self.set_font("Arial", '', 12)
        for label, value in data.items():
            self.cell(0, 10, f"{label}: {value}", ln=True)

    def add_image(self, image_path, w=100):
        if os.path.exists(image_path):
            self.image(image_path, w=w)
        else:
            self.cell(0, 10, "Image not found", ln=True)


def generate_pdf_report(filename, roi_area, force, torn, image_path=None, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    pdf = SimulationReport()
    pdf.add_page()

    data = {
        "Filename": filename,
        "ROI Area": f"{roi_area} pixels",
        "Max Applied Force": f"{force:.2f} N",
        "Tearing Status": "TORN" if torn else "INTACT"
    }

    pdf.add_simulation_data(data)

    if image_path:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Simulation Image:", ln=True)
        pdf.add_image(image_path)

    report_name = f"{filename}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(output_dir, report_name)
    pdf.output(output_path)
    print(f"✅ PDF report saved to: {output_path}")
