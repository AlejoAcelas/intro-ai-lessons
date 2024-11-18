# %%
import pdfplumber

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Usage
pdf_path = "qualifying_round_2024_test.pdf"
text = extract_text_from_pdf(pdf_path)
print(text)
# %%
