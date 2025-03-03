# Run server using: uvicorn src.backend:app --reload
import pprint
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from query_document import QueryDocument
from housing_search import config
return_flag = config['return_flag']


# Initialize FastAPI app
app = FastAPI()

# Request model for API
class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search_documents(request: QueryRequest):
    query_text = request.query

    print(f"🔍 Received search query: {query_text}")
    query_text = [query_text]
    query = QueryDocument()
    if return_flag == 1:
        ss_results, bm_25_results = query.search_and_print_results(query_text, return_flag)
    else:
        final_results = query.search_and_print_results(query_text, return_flag)
    query.db_helper.close()

    print(f"✅ Returning dummy results for query: {query_text}")
    if return_flag == 1:
        return {"query": query_text, "ss_search": [ss_results], "bm25_results": [bm_25_results]}
    
    return {"query": query_text, "Final_search_results": [final_results]}
