import os
import time
from dotenv import load_dotenv
from pydantic import SecretStr 

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Validate Pinecone API key
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if pinecone_api_key is None:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

# Validate Pinecone index name
index_name = os.environ.get("PINECONE_INDEX_NAME")
if index_name is None:
    raise ValueError("PINECONE_INDEX_NAME environment variable is not set.")

# Validate OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Wrap the API key in SecretStr
openai_api_key = SecretStr(openai_api_key)

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists, and create it if it doesn't
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load documents from the 'documents' folder
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.split_documents(raw_documents)

# Generate unique UUIDs for the documents
uuids = [f"id{i+1}" for i in range(len(documents))]

# Add documents to the vector store
vector_store.add_documents(documents=documents, ids=uuids)
