from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Hello, this is a PDF created with Python!", ln=True, align='C')
pdf.output("output.pdf")
print("PDF created: output.pdf")
