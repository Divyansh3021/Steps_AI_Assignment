# Steps AI: NLP Engineer Task

This repository contains a set of scripts designed to scrape NVIDIA CUDA documentation, chunk the content semantically, store embeddings in Milvus, and implement a hybrid retriever using DPR and BM25. It also provides functionality to expand queries and generate answers using Google Generative AI.

## Installation

1. Clone this repository:

```sh
git clone https://github.com/Divyansh3021/Steps_AI_Assignment.git
cd Steps_AI_Assignment
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

## Running the code

In your terminal, type:
```sh
streamlit run app.py
```

## Classes and Functions

### `NvidiaDocsSpider`

#### `__init__() -> None`
Initializes the `NvidiaDocsSpider` class with allowed domains, start URLs, and a set to track visited URLs.

#### `parse(url: str, depth: int) -> list`
Parses the given URL, scrapes the content, and follows sub-links up to a specified depth. Returns a list of dictionaries containing URLs and their content.

#### `is_allowed_domain(url: str) -> bool`
Checks if the given URL is within the allowed domains. Returns a boolean.

#### `run() -> None`
Starts the scraping process from the start URLs, saves the scraped data to `nvidia_docs.json`.

### Chunking and Storing Embeddings

#### `GoogleGenerativeAIEmbeddings`
Initializes Google Generative AI Embeddings with a specified model and API key.

#### `SemanticChunker`
Chunks the scraped data using semantic similarity. Stores chunks along with their embeddings in `nvidia_chunks.json`.

### Milvus Integration

#### `connections.connect() -> None`
Connects to the Milvus server.

#### `Collection`
Defines a collection schema with fields for URL, content, and embedding. Inserts the chunks into the collection.

### Query Expansion

#### `query_expansion(query: str) -> str`
Expands a given query using Google Generative AI. Returns the expanded query.

### Dense Passage Retrieval (DPR)

#### `encode_query(query: str) -> torch.Tensor`
Encodes the given query using DPR Question Encoder. Returns the query embedding.

#### `retrieve_passages(query_embedding: torch.Tensor, embeddings: torch.Tensor, passages: list, top_k: int = 2) -> list`
Retrieves the top passages based on similarity to the query embedding. Returns a list of top passages.

### BM25 Retrieval

#### `BM25Okapi`
Initializes BM25 with the tokenized passages.

#### `retrieve_passages_bm25(query: str, passages: list, top_k: int = 3) -> list`
Retrieves the top passages using BM25. Returns a list of top passages with scores.

### Hybrid Retriever

#### `hybrid_retriever(query: str, embeddings: torch.Tensor, passages: list, top_k: int = 3, alpha: float = 0.5) -> list`
Combines DPR and BM25 results to retrieve top passages. Returns a sorted list of top passages based on combined scores.

### Answer Generation

#### `generate_answer(query: str, context: list) -> str`
Generates an answer to the given query based on the provided context using Google Generative AI. Prints the answer.

## Files

- `nvidia_docs.json`: Contains the scraped NVIDIA documentation.
- `nvidia_chunks.json`: Contains the semantically chunked data with embeddings.

Ensure that the necessary API keys and configurations are set up correctly before running the scripts. The code relies on Google Generative AI, SentenceTransformers, Milvus, and Hugging Face Transformers for various functionalities.

## Note:
1. The `.env` file was pushed deliberately and not by mistake.
2. The `max_depth` denotes the maximum depth till which webscraping will be done, by default its value is 5, but due to hardware limitations, its value was taken 1 during runtime on my machine.