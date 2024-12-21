from pdf2image import convert_from_path
import pytesseract
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfWriter, PdfReader
from io import BytesIO


def ocr_pdf(pdf_file):
    # Converting PDF to images
    images = convert_from_path(pdf_file)

    # Output directory
    output_dir = "ocr_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_pdf_path = os.path.join(
        output_dir, f"{os.path.splitext(os.path.basename(pdf_file))[0]}_searchable.pdf"
    )

    # Create a PDF writer to create the final searchable PDF
    pdf_writer = PdfWriter()

    total_pages = len(images)
    print(f"Starting OCR for {total_pages} pages...")

    for i, image in enumerate(images):
        page_number = i + 1
        print(f"Processing page {page_number} of {total_pages}...")

        # Perform OCR to extract text
        text = pytesseract.image_to_string(image)

        # Create a PDF wit the extracted text
        packet = BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        can.drawString(10, 100, text)
        can.save()

        # Move to the beginning of the StringIO buffer
        packet.seek(0)

        # Create a PDF reader object from the image
        new_pdf = PdfReader(packet)
        image_pdf = PdfReader(BytesIO(image.tobytes()))

        # Combine the image with the OCR text overlay
        page = image_pdf.pages[0]
        page.merge_page(new_pdf.pages[0])

        # Add the page to the final PDF
        pdf_writer.add_page(page)

        print(f"OCR complete for page {page_number}")

    # Write the final searchable PDF
    with open(output_pdf_path, "wb") as output_pdf:
        pdf_writer.write(output_pdf)

    print(f"OCR process completed. Searchable PDF saved as {output_pdf_path}")


pdf_file = "winter24-graduation-dom.pdf"
ocr_pdf(pdf_file)
