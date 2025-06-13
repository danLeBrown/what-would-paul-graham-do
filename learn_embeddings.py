import os
import chromadb
from chromadb.utils import embedding_functions
import time

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

# check if file exists
if not os.path.exists(processed_file):
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    with open(processed_file, "w") as f:
        f.write("[]")  # Initialize with an empty JSON array
else:
    with open(processed_file, "r") as f:
        processed_list = f.read().strip("[]").split(",") if f.read() else []

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pg_essays")


from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# character_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
# )

print(f"Processing files in directory: {directory_path}")

for index, filename in enumerate(
    os.listdir(directory_path)
):  # Limit to first 10 files for testing
    # check if file is already processed
    if filename not in processed_list and filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

            # character_split_texts = character_splitter.split_text("\n\n".join())
            texts = text_splitter.split_text(content)

            token_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=0, tokens_per_chunk=256
            )

            token_split_texts = token_splitter.split_text("\n\n".join(texts))

            # Store each chunk in the collection
            for content in token_split_texts:
                # Add the chunk to the collection with metadata
                # Using filename as the ID for simplicity
                # Ensure the content is not empty
                if not content.strip():
                    continue
                # Add the chunk to the collection
                # Using filename as the ID for simplicity
                collection.add(
                    documents=[content.strip()],
                    metadatas=[{"filename": filename}],
                    ids=[filename],
                )
        # Move the processed file to the processed directory
        processed_list.append(filename)
        with open(processed_file, "w") as f:
            f.write(str(processed_list))
        # Ensure the processed file is updated
        processed_list = list(set(processed_list))
        print(
            f"Processed {filename} and added {len(token_split_texts)} chunks to the collection."
        )
        if index % 10 == 0:
            print(f"Processed {index} files so far...")
            # Sleep to avoid overwhelming the system
            time.sleep(0.2)
print("All files processed and added to the collection.")
