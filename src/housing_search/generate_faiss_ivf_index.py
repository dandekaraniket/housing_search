#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import os
from svlearn_faiss import config
from svlearn_faiss.faiss_common import (compute_embeddings, 
                                        create_faiss_ivf_index, 
                                        load_faiss_index, 
                                        read_jsonl_file, 
                                        save_faiss_index)

# Path to file containing chunks 
chunks_file = config["paths"]["chunks_file"]

# Path to folder containing results that we write into (for eg Faiss index file)
results_dir = config["paths"]["results"]

if __name__ == "__main__":
    faiss_index_file = f"{results_dir}/faiss_index_ivf.bin"
    # Main execution
    if not os.path.exists(faiss_index_file):
        print("Building FAISS index")
        # If no saved index, process chunks file and build FAISS index
        texts = read_jsonl_file(chunks_file)
        embeddings = compute_embeddings(texts)
        faiss_index = create_faiss_ivf_index(embeddings)
        save_faiss_index(faiss_index, faiss_index_file)  # Save for later
    else:
        # Load saved FAISS index
        print("Loading existing FAISS index...")
        faiss_index = load_faiss_index(faiss_index_file)
        
