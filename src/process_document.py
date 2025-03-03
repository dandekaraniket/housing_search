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

import torch, gc
import time

DEVICE = config["device"] 
MODEL = config["model"]
DB_FILE = config["paths"]["db_file"]
tokenizer = AutoTokenizer.from_pretrained(MODEL)
embedder = SentenceTransformer(MODEL).to(DEVICE)

data_dir = config["paths"]["data_dir"]
output_dir = config["paths"]["output_dir"]
faiss_index_file = config["paths"]["faiss_index_file"]
db_helper = db_search.DBHelper(DB_FILE)
document_data_to_enter_in_db = []
chunk_data_to_enter_in_db = []

class DocumentProcessor:
    def __init__(self, embedder, faiss_index_file, output_dir):
        self.embedder = embedder
        self.faiss_index_file = faiss_index_file
        self.output_dir = output_dir
        self.db_helper = db_search.DBHelper(DB_FILE)
        self.db_helper.connect()
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

    def chunk_document(self, doc):
        chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
            enforce_max_tokens=True,
            merge_undersized_chunks=True, 
            max_tokens=256
        )

        chunk_iter = chunker.chunk(dl_doc=doc)
        return chunk_iter

    def serialize_chunk(self, chunk, chunker):
        return chunker.serialize(chunk=chunk)

    def process_document(self, pdf_path, doc_id, all_parts_of_pdf):
        json_file_name = os.path.join(self.output_dir, f"{os.path.basename(pdf_path)}.json")

        if self.check_file_exists(json_file_name) and os.path.getsize(json_file_name) > 0:
           print("this document is already chunked and processed. Skipping.")
           return False
        
        document_data_to_enter_in_db.append({"filename": pdf_path, "doctype": "pdf", "text": "", "id": doc_id})
        try:
            with open(os.path.join(self.output_dir, f"{os.path.basename(pdf_path)}.json"), "w", encoding="utf-8") as f:
                for pdf_split_path_list in all_parts_of_pdf:
                    if len(pdf_split_path_list[0]) > 50:
                           pdf_split_path_list=pdf_split_path_list[0]
                           pdf_split_path_list=pdf_split_path_list.split(',')
                    for pdf_split_path in pdf_split_path_list:
                      pdf_split_path = pdf_split_path.lstrip()
                      doc = self.convert_pdf_to_document(pdf_split_path)
                      if doc is None:
                          return
                    
                      chunk_iter = self.chunk_document(doc)
                            
                      for i, chunk in tqdm(enumerate(chunk_iter), desc=f"Chunking {os.path.basename(pdf_split_path)}"):
                          enriched_text = self.serialize_chunk(chunk, HybridChunker())
                          chunk_data = {"document_id": doc_id, "chunk_id": self.chunk_id, "text": enriched_text}
                          chunk_data_to_enter_in_db.append({"text": enriched_text, "document_id": doc_id, "id": self.chunk_id, "embedding": self.chunk_id})
                          self.chunk_id = self.chunk_id + 1
                          f.write(json.dumps(chunk_data) + '\n')
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
        
        return True

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


def process_pdf(pdf_path, doc_id, processor, queue, all_parts_of_pdf):
    processor.process_document(pdf_path, doc_id, all_parts_of_pdf)
    queue.put(pdf_path) 


def main():
    processor = DocumentProcessor(embedder, faiss_index_file, output_dir)
    db_helper = db_search.DBHelper(DB_FILE)
    db_helper.connect()
    #Place where filenames need to come where Chunked = 0 currently hardcoded in db_search.py&&&
    filenames = db_helper.query_document_splitter_table("document_splitter")
    
    queue = Queue()
    threads = []
    i = 1
    batch_size = 1
    batch_quotient, batch_remainder = divmod(len(filenames), batch_size)
    total_batches = batch_quotient + 1
    for batch in range(1, total_batches + 1):
        gc.collect()
        torch.cuda.empty_cache()
        for file_number in range(i, i + batch_size):
            if i <= len(filenames):
                pdf_path = os.path.join(data_dir, filenames[i-1][0])
                all_parts_of_pdf = db_helper.query_document_splitter_table_splits("document_splitter", pdf_path)
                thread = threading.Thread(target=process_pdf, args=(pdf_path, i, processor, queue, all_parts_of_pdf))
                threads.append(thread)
                thread.start()
                i = i + 1
        
        for thread in threads:
            thread.join()

        while not queue.empty():
            print(f"Processed: {queue.get()}")
        
    #place where we check if chunking is successful for all the files in a batch and then go
    #and update the chunked column in document_splitter to 1&&&

    filenames = db_helper.query_document_splitter_table("document_splitter") 
    for file in filenames:
        json_file_name = os.path.join(processor.output_dir, f"{os.path.basename(file[0])}.json")
        if processor.check_file_exists(json_file_name) and os.path.getsize(json_file_name) > 0:
            db_helper.update_table("document_splitter", "Chunked", 1, "document_name",  file[0])

    #processor.db_helper.close()

if __name__ == "__main__":
    main()

