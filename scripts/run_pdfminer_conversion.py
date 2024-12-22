from pdfminer.high_level import extract_text
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python test_pdfminer.py <pdf_path>")
    sys.exit(1)

pdf_path = sys.argv[1]
base_path = os.path.splitext(pdf_path)[0]
output_path = f"{base_path}_pdfminer.txt"

# Extract text from PDF
text = extract_text(pdf_path)

# Write text to file
with open(output_path, "w") as f:
    f.write(text)

print(f"Converted text saved to {output_path}")
