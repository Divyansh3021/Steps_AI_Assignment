import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
import os

load_dotenv()

class NvidiaDocsSpider:
    def __init__(self):
        self.allowed_domains = ["docs.nvidia.com"]
        self.start_urls = ["https://docs.nvidia.com/cuda/"]
        self.visited_urls = set()
        self.chunks = []
        self.max_depth = 5

    def parse(self, url, depth):
        if url in self.visited_urls or depth > self.max_depth:
            return []
        self.visited_urls.add(url)
        print(f"Scraping: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_content = soup.get_text()

        data = [{
            'url': url,
            'content': page_content
        }]

        # Find and follow sub-links
        if depth < self.max_depth:
            for link in soup.find_all('a', href=True):
                next_url = urljoin(url, link['href'])
                if self.is_allowed_domain(next_url):
                    data.extend(self.parse(next_url, depth + 1))

        return data

    def is_allowed_domain(self, url):
        return any(domain in url for domain in self.allowed_domains)

    def run(self):
        data = []
        for url in self.start_urls:
            data.extend(self.parse(url, 0))
        with open('srcape_docs.json', 'w') as f:
            json.dump(data, f)

    def generate_embeddings(self):

        # Initialize Google Generative AI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.environ['GOOGLE_API_KEY'],
            task_type="retrieval_document"
        )

        # Load the scraped data
        with open('scrape_docs.json') as f:
            data = json.load(f)

        # Initialize Semantic Chunker
        splitter = SemanticChunker(embeddings=embeddings)

        # Chunk data
        for entry in data:
            if entry['content'].strip():  # Ensure the content is not empty
                docs = splitter.create_documents([entry['content']])
                for doc in docs:
                    content = doc.page_content.strip()
                    if content:  # Ensure each chunk is not empty
                        self.chunks.append({
                            'url': entry['url'],
                            'content': content,
                            'embedding': embeddings.embed_query(content)
                        })

        # Save the chunks
        with open('scrape_chunks.json', 'w') as f:
            json.dump(self.chunks, f)


from milvus import default_server
from pymilvus import connections
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

class Storage:
    def __init__(self) -> None:
        default_server.start()

        connections.connect(host='127.0.0.1',port=default_server.listen_port)
    
    def store(self, chunks):

        fields = [
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length =65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        collection_schema = CollectionSchema(fields=fields, schema="DenseVector")
        collection_name_ivf = "ivf_embeddings"

        # Define IVF parameters
        nlist = 128
        metric = "L2" 

        collection = Collection(name=collection_name_ivf, schema=collection_schema, use_index="IVF_FLAT", params={"nlist": nlist, "metric": metric})

        entity = []
        for chunk in chunks:
            dic = {}
            dic['url'] = chunk['url']
            dic['content'] = chunk['content']
            dic['embedding'] = chunk['embedding']
            entity.append(dic)

        collection.insert(entity)


import google.generativeai as genai
import json
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from rank_bm25 import BM25Okapi

class Retriever:
    def __init__(self) -> None:
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        # Load passages and embeddings from JSON
        with open('nvidia_chunks.json', 'r') as f:
            data = json.load(f)

        # Extract passages and embeddings into separate lists
        self.passages = [entry['content'] for entry in data]
        self.embeddings = torch.tensor([entry['embedding'] for entry in data])

        # Initialize BM25
        self.bm25 = BM25Okapi([passage.split() for passage in self.passages])

        # Load DPR models and tokenizers
        self.context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.llm = genai.GenerativeModel('models/gemini-1.5-pro')
        self.modified_query = ""
    
    def query_expansion(self,query):
        prompt = f"""
        System: You are a helpful expert technical research assistant. Provide an example answer to the given question, that might be found in a document like an web scraped data. 
        
        User: {query}
        """

        self.modified_query = self.llm.generate_content(prompt).text
    


    def encode_query(self,query):
        inputs = self.context_tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            query_embedding = self.context_encoder(**inputs).pooler_output
        return query_embedding

    def retrieve_passages_dpr(self, query_embedding, embeddings, passages, top_k=3):
        similarities = torch.matmul(query_embedding, embeddings.T).squeeze(0)
        top_k_indices = torch.topk(similarities, k=top_k).indices
        return [(passages[idx], similarities[idx].item()) for idx in top_k_indices]

    def retrieve_passages_bm25(self, query, passages, top_k=3):
        bm25_scores = self.bm25.get_scores(query.split())
        top_k_indices = torch.topk(torch.tensor(bm25_scores), k=top_k).indices
        return [(passages[idx], bm25_scores[idx]) for idx in top_k_indices]

    def hybrid_retriever(self, query, embeddings, passages, top_k=3, alpha=0.5):
        query_embedding = self.encode_query(query)

        assert query_embedding.shape[1] == embeddings.shape[1], f"Query embedding size {query_embedding.shape} does not match passage embedding size {embeddings.shape}"

        dpr_results = self.retrieve_passages_dpr(query_embedding, embeddings, passages, top_k)
        bm25_results = self.retrieve_passages_bm25(query, passages, top_k)

        # Combine DPR and BM25 results
        combined_scores = {}
        for passage, score in dpr_results:
            combined_scores[passage] = combined_scores.get(passage, 0) + alpha * score
        for passage, score in bm25_results:
            combined_scores[passage] = combined_scores.get(passage, 0) + (1 - alpha) * score

        sorted_passages = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_passages[:top_k]
    
    def return_context(self, query):
        self.query_expansion(query)
        context = self.hybrid_retriever(self.modified_query, self.embeddings, self.passages)
        return context

    def generate_answer(self, query):
        context = self.return_context(query)

        prompt = f"""
        You are given this context and a query, based on the context and your own knowledge, answer the query.

        Context:
        {context}

        Query:
        {query}
        """

        answer = self.llm.generate_content(prompt).text
        return answer




retr = Retriever()