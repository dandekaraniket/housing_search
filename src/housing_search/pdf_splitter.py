import fitz  # PyMuPDF
import os
from housing_search import config

def split_pdf(input_pdf, output_folder, pages_per_split=10):
    doc = fitz.open(input_pdf)
    total_pages = len(doc)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(0, total_pages, pages_per_split):
        output_pdf = os.path.join(output_folder, f"split_{i+1}_{min(i+pages_per_split, total_pages)}.pdf")
        new_doc = fitz.open()
        for j in range(i, min(i + pages_per_split, total_pages)):
            new_doc.insert_pdf(doc, from_page=j, to_page=j)
        new_doc.save(output_pdf)
        new_doc.close()
        print(f"Created: {output_pdf}")

    doc.close()

def find_pdf_files_recursive(data_dir):
    pdf_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".pdf") or file.lower().endswith(".json"):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    return pdf_files

def main():
    output_dir = config["paths"]["split_pdf_dir"]
    data_dir = config["paths"]["data_dir"]
    pdf_files = find_pdf_files_recursive(data_dir)
    for file_name in pdf_files:
        split_pdf(file_name, output_dir)

if __name__ == "__main__":
    main()