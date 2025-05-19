from report_generator import generate_pdf_report

generate_pdf_report(
    filename="test_brain.dcm",
    roi_area=5421,
    force=35.2,
    torn=True,
    image_path="assets/annotated_image.jpg"  # Optional image
)
