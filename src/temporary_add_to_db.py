import fitz  # PyMuPDF
import os
from housing_search import config
from housing_search import db_search
output_dir = config["paths"]["output_dir"]

db_helper = db_search.DBHelper("/home/data/proj1-results/housing_search.db")

def main():
    db_helper.connect()
    names = db_helper.query_document_splitter_table("document_splitter")
    for name in names:
        json_file_name = os.path.join(output_dir, f"{os.path.basename(name[0])}.json")
        if os.path.exists(json_file_name) and os.path.getsize(json_file_name) > 0:
           db_helper.update_table("document_splitter", "Chunked", 1, "document_name",  name[0])
        else:
            db_helper.update_table("document_splitter", "Chunked", 0, "document_name",  name[0])
    db_helper.close()
if __name__ == "__main__":
    main()
