#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import json
from typing import List
import faiss
import numpy
from sentence_transformers import SentenceTransformer
from housing_search import config

# Initialize embedding model (using Sentence Transformers)
model_name = config["model"]
embedding_model = SentenceTransformer(model_name)

# Read JSONL files and extract text
def read_jsonl_file(file_name: str) -> List[str]:
    """Read chunks file and returns the list of text chunks under the text field

    Args:
        file_name (str): filename

    Returns:
        List[str]: The returned list of chunk texts
    """
    # List to store vectors and metadata
    texts = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])  # Adjust key if necessary            
    return texts



# Compute embeddings
def compute_embeddings(text_list: List[str]) -> numpy.ndarray:
    """Computes embeddings of the list of text elements using a sentence transformer

    Args:
        text_list (List[str]): Input list of strings

    Returns:
        numpy.ndarray: embeddings
    """
    return embedding_model.encode(text_list, convert_to_numpy=True, show_progress_bar=True)

# Build FAISS L2 index
def create_faiss_l2_index(embeddings: numpy.ndarray) :
    """Creates the FAISS L2 index

    Args:
        embeddings (numpy.ndarray): embeddings

    Returns:
        _type_: Faiss L2 index
    """
    dimension = embeddings.shape[1]  # Get embedding dimension
    index = faiss.IndexFlatL2(dimension)  # L2 similarity
    index.add(embeddings)  # Add vectors to index
    return index

# Build FAISS HNSW index
def create_faiss_hnsw_index(embeddings:numpy.ndarray, hnsw_m: int=32):
    """    Creates an HNSW index with M neighbors per node.
    Higher M leads to better recall but increases memory and indexing time.

    Args:
        embeddings (numpy.ndarray): embeddings to be indexed
        hnsw_m (int, optional): the M parameter. Defaults to 32.
    Returns:
        _type_: Faiss HNSW index
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, hnsw_m)  # HNSW with L2 distance
    index.hnsw.efConstruction = 200  # Control search quality (higher = better recall)
    index.add(embeddings)  # Add vectors to the index
    return index

# Build FAISS IVF index
def create_faiss_ivf_index(embeddings: numpy.ndarray, nlist: int=100):
    """Creates an IVF index with flat quantization.
    nlist: number of clusters (higher nlist means better search quality, but more memory).

    Args:
        embeddings (numpy.ndarray): embeddings to be indexed
        nlist (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: Faiss IVF index
    """
    dimension = embeddings.shape[1]
    print(dimension)
    
    # Create the IVF index
    quantizer = faiss.IndexFlatL2(dimension)  # The quantizer (Flat L2 index)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)  # IVF with L2 distance
    
    # Train the index with a subset of embeddings
    index.train(embeddings)  # Training step to generate clusters
    index.add(embeddings)  # Add data to the index
    return index

# Save FAISS index
def save_faiss_index(index, filename):
    faiss.write_index(index, filename)  # Save index to disk

# Load FAISS index
def load_faiss_index(filename):
    return faiss.read_index(filename)  # Load index from disk

# Search FAISS L2 index (or FAISS IVF index)
def search_l2_index(query, index, k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# Search FAISS HNSW index
def search_hnsw_index(query, index, k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    index.hnsw.efSearch = 128  # Adjust tradeoff between speed and recall
    distances, indices = index.search(query_embedding, k)
    return distances, indices
