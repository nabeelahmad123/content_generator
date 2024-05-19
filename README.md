# RAG System using LLamaIndex for Questioning PDFs

This project implements a Retrieval-Augmented Generation (RAG) system utilizing LLamaIndex to perform question answering on PDF documents. The application leverages the power of Large Language Models (LLMs) to generate contextually relevant responses to user queries based on the content of the PDFs.

## Features

- **Advanced LLM Integration**: Utilizes LLamaIndex with a powerful LLM for sophisticated content retrieval and generation.
- **Question Answering**: Allows users to ask questions about the content of PDFs and get precise answers.
- **Document Parsing and Indexing**: Efficiently parses and indexes PDF documents to enable fast and accurate querying.
- **Similarity-Based Retrieval**: Employs similarity measures to retrieve the most relevant pieces of information from the indexed documents.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/nabeelahmad123/RAG_System_using_LLamaIndex.git
    cd RAG_System_using_LLamaIndex
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key:
    - Create a `.env` file in the root directory of your project and add your OpenAI API key:
      ```
      OPENAI_API_KEY=your_openai_api_key
      ```

4. Ensure you have the required model files:
    - Download the necessary model files and place them in the appropriate directory as specified in the code.

## Usage

1. Ensure your PDF files are in the `data` directory.

2. Run the Jupyter notebook:
    ```sh
    jupyter notebook PDF_QA.ipynb
    ```

3. Follow the steps in the notebook to parse the PDFs, create the index, and query the system.

## How It Works

This project uses LLamaIndex to parse and index PDF documents. The indexing process converts the document content into vector representations, which allows for efficient retrieval of relevant text based on user queries. The integration of LLMs enables the system to generate human-like responses that are contextually relevant to the queried information.

### Code Overview

```python
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load API key from .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Load and index documents
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# Create query engine
query_engine = index.as_query_engine()

# Example query
response = query_engine.query("What are LLMs?")
print(response)

# Advanced query with postprocessing
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.80)
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

response = query_engine.query("Does LangChain create fast applications?")
print(response)

# Pretty print response
from llama_index.core.response.pprint_utils import pprint_response
pprint_response(response, show_source=True)
print(response)

# Persistent storage for index
import os.path
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Query the persistent index
query_engine = index.as_query_engine()
response = query_engine.query("What are transformers?")
print(response)
