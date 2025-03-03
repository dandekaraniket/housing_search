import json
import os
import threading
from housing_search import config
from tqdm import tqdm
from housing_search import db_search
from queue import Queue
import pickle
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
from housing_search.faiss_common import create_faiss_ivf_index, save_faiss_index, \
    read_jsonl_file, load_faiss_index, search_l2_index, create_faiss_hnsw_index
from transformers import AutoTokenizer, AutoModel

DEVICE = config["device"] 
MODEL = config["model"]
DB_FILE = config["paths"]["db_file"]
BM25_CORPUS_FILE = config['paths']["bm25_corpus_file"]

data_dir = config["paths"]["data_dir"]
output_dir = config["paths"]["output_dir"]
faiss_index_file = config["paths"]["faiss_index_file"]
db_helper = db_search.DBHelper(config["paths"]["db_file"])


class BM25DocumentProcessor:
    def __init__(self):
        #self.output_dir = output_dir
        self.db_helper = db_search.DBHelper(DB_FILE)
        self.db_helper.connect()
        
    def create_bm25_passages(self):
      self.passages = []
      results = self.db_helper.query_data("chunk")
      for result in results : 
          # Parse each line into a dictionary
          text = result['text'].strip()
          chunk_id = result['embedding']
          passage = f"{text}" if chunk_id else text
          self.passages.append(passage)
      return self.passages

    def bm25_tokenizer(self, text):
      tokenized_doc = []
      for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
      return tokenized_doc

    def create_and_return_bm25_corpus_tokens(self):
      passages = self.create_bm25_passages()
      self.tokenized_doc = []
      for passage in passages:
        self.tokenized_doc.append(self.bm25_tokenizer(passage))
      self.bm25 = BM25Okapi(self.tokenized_doc)
      with open(BM25_CORPUS_FILE, 'wb') as f:
        pickle.dump(self.bm25, f)
      return True
    
    def tokenize_query_and_return_results(self, query):
       tokenized_query = self.bm25_tokenizer(query)
       bm25_scores = self.bm25.get_scores(tokenized_query)
       return bm25_scores

def main():
  processor = BM25DocumentProcessor(output_dir) 
  passages = processor.create_bm25_passages()
  tokenized_passages = processor.bm25_tokenizer(passages)
  bm25 = BM25Okapi(tokenized_passages)

if __name__ == "__main__":
    main()
