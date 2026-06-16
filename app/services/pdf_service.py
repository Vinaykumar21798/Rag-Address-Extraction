import fitz

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

