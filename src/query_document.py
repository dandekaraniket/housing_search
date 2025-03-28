from housing_search import config
from tqdm import tqdm
from housing_search import db_search
from create_bm25_corpus import BM25DocumentProcessor
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
#from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from housing_search.faiss_common import create_faiss_ivf_index, save_faiss_index, read_jsonl_file, load_faiss_index, search_l2_index, search_hnsw_index
import pprint
import numpy as np
from IPython.display import display, HTML
import os
import pickle
from langchain.chat_models import ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI
import re   
return_flag = config['return_flag']

class QueryDocument:
    def __init__(self):
        DEVICE = config["device"] 
        MODEL = config["model"]
        RE_RANKER = config["re_ranker"]
        self.num_results = config["num_results"]
        embedder = SentenceTransformer(MODEL).to(DEVICE)
        self.re_ranker_model = AutoModelForSequenceClassification.from_pretrained(RE_RANKER)
        self.re_ranker_tokenizer = AutoTokenizer.from_pretrained(RE_RANKER)
        #self.re_ranker_model = BertForSequenceClassification.from_pretrained(RE_RANKER)
        #self.re_ranker_tokenizer = BertTokenizer.from_pretrained(RE_RANKER)
        output_dir = config["paths"]["output_dir"]
        faiss_index_file = config["paths"]["faiss_index_file"]
        self.BM25_CORPUS_FILE = config['paths']['bm25_corpus_file']
        self.embedder = embedder
        self.faiss_index_file = faiss_index_file
        self.output_dir = output_dir
        self.db_helper = db_search.DBHelper(config["paths"]["db_file"])
    
    def search_and_print_results(self, search_queries, return_flag):
        self.db_helper.connect()
        faiss_ivf_index = load_faiss_index(self.faiss_index_file)
        text_result = []
        for query in search_queries:
            print("*********************************************************\n\n")
            print(f"Nearest neighbors of {query}\n")
            #distances, indices = search_l2_index(query, faiss_ivf_index, k=10)
            query = self.query_rewrite_prompt(query)
            print(f"Rewritten query is::::::: {query}\n")
            distances, indices = search_hnsw_index(query, faiss_ivf_index, k=10)
            for i, idx in enumerate(indices[0]):
                if idx > 0:
                    print(f"[Rank {i+1}]: [Idx: {idx}]  (Distance: {distances[0][i]})")
                    chunk_query_result = self.db_helper.query_data_specific("chunk", idx+1)
                    chunk_query_result_list = list(chunk_query_result)
                    document_name = self.db_helper.query_data_document_table("document", chunk_query_result_list[1])
                    document_pdf = os.path.basename(document_name[0]).replace('.json', '')
                    document_name = self.db_helper.query_document_name_from_splitter_table("document_splitter",document_pdf)
                    chunk_query_result_list[1] = document_name
                    chunk_query_result = tuple(chunk_query_result_list)
                    text_result.append(chunk_query_result)

        bm25_search_results_final = self.search_and_print_bm25_results(search_queries)
        #call bm25 search here
        if return_flag == 1:
            return text_result, bm25_search_results_final
        for each_result in bm25_search_results_final:
            text_result.append(each_result)
        #This line of code will break if search queries is multiple queries in the list
        final_reranked_results = self.rerank_bm25_and_ss_results(text_result, search_queries[0])
        opt_result  = ""
        for result in final_reranked_results:
            opt_result = opt_result + "".join(result[0])
    
        final_opt_result_list = [self.result_rewrite_prompt(opt_result)]
        final_opt_result_list.insert(1,{"Summary Result"})
        final_opt_result = tuple(final_opt_result_list)
        
        final_reranked_results.insert(0,final_opt_result)
        return final_reranked_results

    def search_and_print_bm25_results(self, search_queries):
        for query in search_queries:
            print("*********************************************************\n\n")
            bm25_corpus = BM25DocumentProcessor()
            if os.path.exists(self.BM25_CORPUS_FILE) and os.path.getsize(self.BM25_CORPUS_FILE) > 0:
               with open(self.BM25_CORPUS_FILE, 'rb') as f:
                  bm25_corpus.bm25 = pickle.load(f)
            else :
                bm25_corpus.create_and_return_bm25_corpus_tokens()

            bm25_scores = bm25_corpus.tokenize_query_and_return_results(query)
            top_n = np.argpartition(bm25_scores, -5)[-5:]
            bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
            bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
            bm25_search_results_tuple = ()
            bm25_search_results_final = []
            for hit in bm25_hits[0:3]:
                bm25_search_result = self.db_helper.query_data_specific("chunk", hit['corpus_id']+1)
                #bm25_search_result = bm25_corpus.passages[hit['corpus_id']].replace("\n", " ")
                document_id = self.db_helper.query_data_specific("chunk", hit['corpus_id'])[1]
                document_name = self.db_helper.query_data_document_table("document", document_id)
                bm25_search_results_tuple = (bm25_search_result[0], document_name)
                bm25_search_results_final.append(bm25_search_results_tuple)
                
            return bm25_search_results_final
        
    def rerank_bm25_and_ss_results(self, search_result, search_query):
        scores = {}
        final_search_result = []
        for i, result in enumerate(search_result):
            inputs = self.re_ranker_tokenizer(search_query, result[0], return_tensors="pt", truncation=True, padding=True)
            re_rank_search_result = self.re_ranker_model(**inputs) 
            similarity_score = re_rank_search_result.logits
            scores[similarity_score[0].item()] = i
        hits = sorted(scores.keys(), reverse=True)
        result_count = 1
        for hit in hits:
            final_search_result.append(search_result[scores[hit]])
            result_count += 1
            if result_count > self.num_results:
                break
        return final_search_result
    
    def query_rewrite_prompt(self, query):

        query_rewrite_prompt = PromptTemplate.from_template(
            "Rewrite the following search query to be more specific and optimized for semantic search: {query}"
        )

        client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama',
        )
        
        prompt = query_rewrite_prompt.format(query=query)

        response = client.completions.create(
            model="llama3.2",
            prompt=prompt,
            n=1,
            max_tokens=250,
            temperature=1.0,
            top_p=1.0,
            stop=None,
            stream=False
        )

        optimized_query = response.choices[0].text.strip()

        match = re.findall(r'"(.*?)"', optimized_query, re.DOTALL)
        if match:
            optimized_query = match[0]
        return optimized_query

    def result_rewrite_prompt(self, result):
    
        result_rewrite_prompt = PromptTemplate.from_template(
            f"Based on the following search results, provide a detailed summary: {result}:"
        )      
        client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama',
        )    
        prompt = result_rewrite_prompt.format(result=result)     
        response = client.completions.create(
            model="llama3.2", 
            prompt=prompt,
            n=1,
            max_tokens=1024,
            temperature=0.7,
            top_p=1.0,
            stop=None,
            stream=False
        )   
        optimized_result  = response.choices[0].text.strip()
        match = re.findall(r'"(.*?)"', optimized_result, re.DOTALL)
        if match:
            optimized_result = match[0]
        client._client.close()
        return optimized_result


def main():
    query = QueryDocument()
    search_query = ["Requirements for Fence in Sunnyvale"]
    search_result = [query.search_and_print_results(search_query)]
    bm25_corpus = BM25DocumentProcessor()
    bm25_corpus_tokenized = bm25_corpus.create_and_return_bm25_corpus_tokens()
    bm25_scores = bm25_corpus.tokenize_query_and_return_results(search_query[0], bm25_corpus_tokenized)
    top_n = np.argpartition(bm25_scores, -5)[-5:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    bm25_search_result = []
    for hit in bm25_hits[0:3]:
        bm25_search_result.append(bm25_corpus.passages[hit['corpus_id']].replace("\n", " "))

    scores = []
    pprint.pprint(bm25_search_result)

    pprint.pprint("************************")
    search_result = search_result[0]
    pprint.pprint(search_result)
    search_result.append(bm25_search_result)
    reranked_results = query.rerank_bm25_and_ss_results(search_result, search_query)
    """RE_RANKER = config["re-ranker"]
    re_ranker_model = AutoModelForSequenceClassification.from_pretrained(RE_RANKER)
    re_ranker_tokenizer = AutoTokenizer.from_pretrained(RE_RANKER)


    for result in search_result:
      result=list(result)
      inputs = re_ranker_tokenizer(search_query[0], result[0], return_tensors="pt", truncation=True, padding=True)
      re_rank_search_result = re_ranker_model(**inputs) 
      similarity_score = re_rank_search_result.logits
      scores.append(similarity_score.item())
    for i, score in enumerate(scores):
      print(f"search_result {i+1}: {score}")
      if score > 0:
        #print(f"Re-ranked_result", search_result[i][0])
        print("")"""
    query.db_helper.close()

if __name__ == "__main__":
    main()
