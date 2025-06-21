import getpass
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import numpy as np
from utils import recursively_strip_dict

load_dotenv()

model = None
model_name = os.environ.get("OLLAMA_MODEL", os.environ.get("MODEL_NAME", "gemini-2.0-flash")) 

if os.environ.get("USE_OLLAMA"):
    print("Using Ollama as the model provider.")
    from langchain_ollama.llms import OllamaLLM

    # from langchain.callbacks.manager import CallbackManager
    # from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    # model = OllamaLLM(
    #     model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
    #     # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    # )
    # Initialize the client with Ollama's API
    model = OpenAI(
        base_url="http://localhost:11434/v1",  # Ollama's API endpoint
        api_key="ollama",  # Required but unused (can be any string)
    )
else:
    print("Using Google Gemini as the model provider.")
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(
            "Enter API key for Google Gemini: "
        )
        from langchain.chat_models import init_chat_model

        # model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        model = OpenAI(
            model="gemini-2.0-flash",
            api_key=os.environ.get("GOOGLE_API_KEY"),
            model_provider="google_genai",
        )
        

if model is None:
    raise ValueError(
        "Model could not be initialized. Please check your environment variables."
    )

query = "What are the factors for doing great work?"

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pg_essays")

# results = collection.query(
#     query_texts=query, n_results=10, include=["documents", "metadatas"]
# )

# # print(f"Query: {query}")
# for i, doc in enumerate(results["documents"]):
#     print(f"Result {i + 1}:")
#     print(f"Document: {doc}")
#     print(f"Metadata: {results['metadatas'][i]}")
#     print(f"ID: {results['ids'][i]}")
#     print("-" * 40)

# retrieved_documents = results["documents"][0]

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
    The user query will be provided, and you will generate augmented queries based on it.
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
    
    response = model.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        stream=False,
        response_format={'type': 'json_object'},
        # max_tokens=500,
        # temperature=0.7,
    )
    
    # print("Response from the model:", response)
    queries = {}
    try:
        load_res = json.loads(response.choices[0].message.content)
        print("Loaded response:", load_res)
        if "queries" in load_res and "original_query" in load_res["queries"] and "augmented_queries" in load_res["queries"]:
            queries = load_res
        else:
            # # strip the keys to match the expected structure
            # for key in load_res:
            #     strip_key = key.strip()
            #     if strip_key == "queries":
            #         for k in load_res[key]:
            #             strip_k = k.strip()
            #             if strip_k  == "original_query" or strip_k == "augmented_queries":
            #                 queries[strip_key][strip_k] = load_res[strip_key][strip_k]
            queries= recursively_strip_dict(load_res)
    except json.JSONDecodeError:
        print("Failed to parse JSON response from the model.")
        queries = {}
    return queries

aug_queries = generate_multi_query(query)

joint_query = [
    aug_queries["queries"]["original_query"]
] + aug_queries["queries"]["augmented_queries"]

# print("\nJoint Query:", joint_query)

results = collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "metadatas"]
)
retrieved_documents = results["documents"]
retrieved_metadatas = results["metadatas"]

# # print(retrieved_metadatas[0])

# # print("Retrieved Documents:")
# # for i, doc in enumerate(results["documents"]):
# #     print(f"Result {i + 1}:")
# #     print(f"Document: {doc}")
# #     print(f"Metadata: {results['metadatas'][i]}")
# #     print(f"ID: {results['ids'][i]}")
# #     print("-" * 40)

# # Deduplicate the retrieved documents
# unique_documents = set()
# unique_document_metadata = set()
# enumerated_metadatas = enumerate(retrieved_metadatas)

# for documents in retrieved_documents:
#     for index, document in enumerate(documents):
#         old_count = len(unique_documents)
#         unique_documents.add(document)
#         if len(unique_documents) > old_count:
#             # print(f"Adding unique document: {document}")
#             print(enumerated_metadatas[index]['filename'])
#             unique_document_metadata.add(enumerated_metadatas[index]['filename'])
        
# unique_documents = list(unique_documents)
# unique_document_metadata = list(unique_document_metadata)

# print(f"Retrieved {len(unique_documents)} unique documents.")
# print(f"Unique {len(unique_document_metadata)} unique metadata entries.")


# pairs = []
# for doc in unique_documents:
#     pairs.append([aug_queries["queries"]["original_query"], doc])

# scores = cross_encoder.predict(pairs)

# # print("Scores:")
# # for score in scores:
# #     print(score)

# # print("New Ordering:")
# # for o in np.argsort(scores)[::-1]:
# #     print(o)

# # ====
# top_indices = np.argsort(scores)[::-1][:5]
# top_documents = [unique_documents[i] for i in top_indices]

# # Concatenate the top documents into a single context
# context = ""
# # top_documents = [f"{doc[0]} (from {doc[1]})" for doc in top_documents]
# top_documents = [f"{doc} (from {unique_document_metadata[i]})" for i, doc in enumerate(top_documents)]
# context = "\n\n".join(top_documents)

# Step 1: Get the original query (ensure aug_queries is called)
# augmented_data = aug_queries()  # Assuming this returns {"queries": {"original_query": "..."}}
# original_query = augmented_data["queries"]["original_query"]
original_query = aug_queries["queries"]["original_query"]

# Step 2: Deduplicate documents + metadata
unique_pairs = list({(doc, meta["filename"]): (doc, meta) 
                     for doc, meta in zip(retrieved_documents[0], retrieved_metadatas[0])}.values())
unique_documents = [doc for doc, meta in unique_pairs]

# Step 3: Score documents with Cross-Encoder
pairs = [[original_query, doc] for doc in unique_documents]
scores = cross_encoder.predict(pairs)
top_indices = np.argsort(scores)[::-1][:5]

# Step 4: Format context
context = "\n\n".join(
    f"{doc}\n(Source: {unique_pairs[i][1]['filename']})" 
    for i, doc in enumerate(unique_documents) if i in top_indices
)
print("\nTop Documents:")
print(context)

def generate_answer(query, context, model=model):
    prompt = """
    You are an AI assistant trained on Paul Graham's essays. Use the following ranked excerpts from his works (ordered by relevance) to answer the user's question. Follow these rules:
        1. **Stay faithful to the context**: Do not speculate or add information outside the provided excerpts.
        2. **Prioritize highly ranked excerpts**: Focus on the top 3-5 most relevant passages.
        3. **Be concise**: Summarize key ideas in 1-3 sentences, then quote the most relevant passage verbatim (with citation if available).
        4. **If unclear, say so**: If the context doesn't contain enough information, respond: "Paul Graham hasn't explicitly addressed this, but related ideas include: [summary]."
    Generate your answer in a JSON format with the following structure:
    {
        "summary": "<concise summary of the answer>",
        "quote": "<verbatim quote from the context>",
        "citation": "<citation if available, otherwise empty>"
    }
    """
    
    response = model.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"based on the following context:\n\n{context}\n\nAnswer the query: '{query}'",},
        ],
        stream=False,
        response_format={'type': 'json_object'},
        # max_tokens=500,
        # temperature=0.7,
    )
    
    # print("Response from the model:", response)

    try:
        res = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("Failed to parse JSON response from the model.")
        res = {
            "summary": "Unable to generate a response.",
            "quote": "",
            "citation": ""
        }
    return res

res = generate_answer(query=query, context=context, model=model)
print("Final Answer:")
print(res)

# from utils import word_wrap, project_embeddings

# # output the results documents
# for i, documents in enumerate(retrieved_documents):
#     print(f"Query: {joint_query[i]}")
#     print("")
#     print("Results:")
#     for doc in documents:
#         print(word_wrap(doc))
#         print("")
# print("-" * 100)

# embeddings = collection.get(include=["embeddings"])["embeddings"]
# umap_transform = umap.get(random_state=0, transform_seed=0).fit(embeddings)
# projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)