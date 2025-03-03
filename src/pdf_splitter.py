import fitz  # PyMuPDF
import os
import shutil
from housing_search import config
from housing_search import db_search

db_helper = db_search.DBHelper(config["paths"]["db_file"])

def split_pdf(input_pdf, output_folder, pages_per_split=10):
    doc = fitz.open(input_pdf)
    total_pages = len(doc)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    total_splits_of_pdf = []
    for i in range(0, total_pages, pages_per_split):
        file_name = input_pdf.split('/')[-1]
        output_pdf = os.path.join(output_folder, f"{file_name}_split_{i+1}_{min(i+pages_per_split, total_pages)}.pdf")
        total_splits_of_pdf.append(output_pdf)
        new_doc = fitz.open()
        for j in range(i, min(i + pages_per_split, total_pages)):
            new_doc.insert_pdf(doc, from_page=j, to_page=j)
        new_doc.save(output_pdf)
        new_doc.close()
        print(f"Created: {output_pdf}")
    total_splits_of_pdf = ", ".join(total_splits_of_pdf)
    db_helper.insert_data("document_splitter", {'document_name': input_pdf, "document_splits": total_splits_of_pdf, 'Chunked': 0})
    doc.close()

def find_pdf_files_recursive(data_dir):
    pdf_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".pdf") or file.lower().endswith(".json"):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    return pdf_files

def remove_comma_and_rename(filepath):
    try:
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        new_filename = filename.replace(",", "")

        new_filepath = os.path.join(directory, new_filename)

        # Rename the file
        if filepath != new_filepath: #prevents an error if the filename had no commas.
            shutil.move(filepath, new_filepath)
            print(f"Renamed '{filename}' to '{new_filename}'")
        else:
            print(f"Filename '{filename}' contained no commas. No rename necessary.")

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    db_helper.connect()

    schema1 = """
    document_name TEXT,
    document_splits LONGTEXT,
    Chunked bool DEFAULT 0
    """
    ## Added new chunked bool column&&&
    db_helper.create_table("document_splitter", schema1)
    output_dir = config["paths"]["split_pdf_dir"]
    data_dir = config["paths"]["data_dir"]
    pdf_files = find_pdf_files_recursive(data_dir)
    for file_name in pdf_files:
        remove_comma_and_rename(file_name)
    pdf_files = find_pdf_files_recursive(data_dir)
    for file_name in pdf_files:
        split_pdf(file_name, output_dir)

if __name__ == "__main__":
    main()
