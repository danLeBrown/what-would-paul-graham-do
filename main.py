import getpass
import os
from dotenv import load_dotenv

load_dotenv()

model = None

if os.environ.get("USE_OLLAMA"):
    print("Using Ollama as the model provider.")
    from langchain_ollama.llms import OllamaLLM

    # from langchain.callbacks.manager import CallbackManager
    # from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    model = OllamaLLM(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
else:
    print("Using Google Gemini as the model provider.")
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(
            "Enter API key for Google Gemini: "
        )
        from langchain.chat_models import init_chat_model

        model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

if model is None:
    raise ValueError(
        "Model could not be initialized. Please check your environment variables."
    )

# response = model.invoke("Are you there?")
# print(response)

import chromadb
from chromadb.utils import embedding_functions
# from sentence_transformers import SentenceTransformer

sentence_transformer_model_path = "./locals/sentence_transformers/all-MiniLM-L6-v2"
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    cache_folder=sentence_transformer_model_path,
)
    
if not sentence_transformer_ef:
    raise ValueError(
        "Sentence Transformer embedding function could not be initialized. "
        "Please check the model path or installation."
    )

directory_path = "./pg_essays"

if not os.path.exists(directory_path):
    raise FileNotFoundError(f"Directory {directory_path} does not exist.")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pg_essays")


from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# character_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
# )

# print(f"Processing files in directory: {directory_path}")

# for filename in os.listdir(directory_path)[:10]:  # Limit to first 10 files for testing
#     if filename.endswith(".txt"):
#         file_path = os.path.join(directory_path, filename)
#         with open(file_path, "r", encoding="utf-8") as file:
#             content = file.read()
            
#             # character_split_texts = character_splitter.split_text("\n\n".join())
#             texts = text_splitter.split_text(content)

#             token_splitter = SentenceTransformersTokenTextSplitter(
#                 chunk_overlap=0, tokens_per_chunk=256
#             )
            
#             token_split_texts = token_splitter.split_text("\n\n".join(texts))

#             # Store each chunk in the collection
#             for content in token_split_texts:
#                 # Add the chunk to the collection with metadata
#                 # Using filename as the ID for simplicity
#                 # Ensure the content is not empty
#                 if not content.strip():
#                     continue
#                 # Add the chunk to the collection
#                 # Using filename as the ID for simplicity                
#                 collection.add(
#                     documents=[content.strip()],
#                     metadatas=[{"filename": filename}],
#                     ids=[filename]
#                 )
#             print(f"Processed {filename} and added {len(token_split_texts)} chunks to the collection.")
# print("All files processed and added to the collection.")


query = "How do I tackle imposter syndrome in my career?"

results = collection.query(
    query_texts=query, n_results=10, include=["documents", "metadatas"]
)

# print(f"Query: {query}")
# for i, doc in enumerate(results["documents"]):
#     print(f"Result {i + 1}:")
#     print(f"Document: {doc}")
#     print(f"Metadata: {results['metadatas'][i]}")
#     print(f"ID: {results['ids'][i]}")
#     print("-" * 40)

retrieved_documents = results["documents"][0]

from sentence_transformers.cross_encoder import CrossEncoder

# 1. Load a pretrained CrossEncoder model
cross_encoder_model_path = "./locals/cross-encoder/stsb-distilroberta-base"
cross_encoder = CrossEncoder(
    "cross-encoder/stsb-distilroberta-base",
    cache_folder=cross_encoder_model_path,
)

import json

def generate_multi_query(query, model=model):
    prompt = """
    You are an assistant designed to expand and enrich user queries for a RAG system focused on Paul Graham's essays. Your task is to generate 3-5 alternative or related queries that explore different angles of the original question. These queries should:
        1. Uncover deeper or broader context
        2. Explore related concepts Paul Graham often discusses
        3. Use synonyms or rephrasing of key terms
        4. Maintain the original intent while diversifying perspectives
    The original query is: """ + query + """
    Generate the queries in a JSON format with the following structure:
    {
        "queries": {
            "original_query": "<original query>",
            "augmented_queries": [
                "<augmented query 1>",
                "<augmented query 2>",
                "<augmented query 3>",
                ...
            ]
        }
    }
    """
    
    response = model.invoke(prompt)
    
    print("Response from the model:", response)

    # try:
    #     queries = json.loads(response)["queries"]
    # except json.JSONDecodeError:
    #     print("Failed to parse JSON response from the model.")
    #     queries = []
    # return queries

aug_queries = generate_multi_query(query)

# 1. First step show the augmented queries
# for query in aug_queries:
#     print("\n", query)
