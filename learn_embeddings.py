import os
import chromadb
from chromadb.utils import embedding_functions
import time
import json

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
processed_file = "./processed_pg_essays.json"
processed_list = []

if not os.path.exists(directory_path):
    raise FileNotFoundError(f"Directory {directory_path} does not exist.")

# Load existing processed files
if os.path.exists(processed_file):
    try:
        with open(processed_file, "r") as f:
            processed_list = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        processed_list = []
else:
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    processed_list = []

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pg_essays")

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

print(f"Processing files in directory: {directory_path}")
print(f"Already processed {len(processed_list)} files: {processed_list}")

for index, filename in enumerate(os.listdir(directory_path)):
    # Skip if file is already processed
    if filename in processed_list or not filename.endswith(".txt"):
        continue
        
    file_path = os.path.join(directory_path, filename)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

            texts = text_splitter.split_text(content)

            token_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=0, tokens_per_chunk=256
            )

            token_split_texts = token_splitter.split_text("\n\n".join(texts))

            # Store each chunk in the collection
            chunk_count = 0
            for i, chunk_content in enumerate(token_split_texts):
                # Ensure the content is not empty
                if not chunk_content.strip():
                    continue
                    
                # Create unique ID for each chunk
                chunk_id = f"{filename}_{i}"
                
                collection.add(
                    documents=[chunk_content.strip()],
                    metadatas=[{"filename": filename, "chunk_index": i}],
                    ids=[chunk_id],
                )
                chunk_count += 1

        # Add to processed list and save immediately
        processed_list.append(filename)
        with open(processed_file, "w") as f:
            json.dump(processed_list, f, indent=2)
        
        print(f"Processed {filename} and added {chunk_count} chunks to the collection.")
        
        if index % 10 == 0:
            print(f"Processed {index} files so far...")
            time.sleep(0.2)
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        if "MPS backend out of memory" in str(e):
            break  # Stop processing if out of memory
        continue

print("All files processed and added to the collection.")
print(f"Total processed files: {len(processed_list)}")