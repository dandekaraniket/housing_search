import json
import os
import threading
from housing_search import config
from tqdm import tqdm
from housing_search import db_search
from queue import Queue
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from housing_search.faiss_common import create_faiss_ivf_index, save_faiss_index, \
    read_jsonl_file, load_faiss_index, search_l2_index, create_faiss_hnsw_index
from transformers import AutoTokenizer, AutoModel

DEVICE = config["device"] 
MODEL = config["model"]
DB_FILE = config["paths"]["db_file"]
tokenizer = AutoTokenizer.from_pretrained(MODEL)
embedder = SentenceTransformer(MODEL).to(DEVICE)

data_dir = config["paths"]["data_dir"]
output_dir = config["paths"]["output_dir"]
faiss_index_file = config["paths"]["faiss_index_file"]
db_helper = db_search.DBHelper(config["paths"]["db_file"])
document_data_to_enter_in_db = []
chunk_data_to_enter_in_db = []


class DocumentDBOperator:
    def __init__(self, embedder, faiss_index_file, output_dir):
        self.embedder = embedder
        self.faiss_index_file = faiss_index_file
        self.output_dir = output_dir
        self.db_helper = db_search.DBHelper(DB_FILE)
        self.chunk_id = 1

    def create_db_schemas(self):
        if self.db_helper.connect():
            schema1 = """
                filename TEXT,
                doctype VARCHAR(64),
                text LONGTEXT,
                id INT(11)
            """
            self.db_helper.create_table("document", schema1)

            schema2 = """
                text TEXT,
                embedding LONGTEXT,
                document_id INT(11),
                id INT(11)
            """
            self.db_helper.create_table("chunk", schema2)
        else:
            print("Failed to connect to the database.")

    def check_file_exists(self, file_path):
        if os.path.exists(file_path):
            return True
        else:
            return False

    def convert_pdf_to_document(self, pdf_path):
        try:
            print(f"Creating Docling for Document {pdf_path}")
            converter = DocumentConverter()
            doc = converter.convert(source=pdf_path).document
            return doc
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return None
    
    def create_and_index_embeddings(self, total_documents):
        total_text = []
        texts = [row['text'] for row in self.db_helper.query_data("chunk")]   
        total_text = texts
        print(f"Creating Embeddings")
        embeddings = self.embedder.encode(total_text, convert_to_numpy=True, show_progress_bar=True)
        faiss_index = create_faiss_hnsw_index(embeddings)
        save_faiss_index(faiss_index, self.faiss_index_file)

def find_pdf_files_recursive(data_dir):
    pdf_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".pdf") or file.lower().endswith(".json"):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    return pdf_files
    
def main():
    processor = DocumentDBOperator(embedder, faiss_index_file, output_dir)
    processor.create_db_schemas()

    chunked_json_files = find_pdf_files_recursive(output_dir)
    document_id = 1
    rewritten_chunk_id = 1
    for file_name in chunked_json_files:
        processor.db_helper.insert_data("document", {"filename": file_name, "doctype": "pdf", "text": "", "id": document_id})
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data:
                    data_2_enter_in_db = {"text": data['text'], "document_id": document_id, "id": rewritten_chunk_id, "embedding": rewritten_chunk_id}
                    processor.db_helper.insert_data("chunk", data_2_enter_in_db)
                    rewritten_chunk_id = rewritten_chunk_id + 1
        document_id = document_id + 1
        

    processor.create_and_index_embeddings(len(chunked_json_files))
    
    processor.db_helper.close()

if __name__ == "__main__":
    main()
