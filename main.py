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

query = "Good artists copy; great artists steal. What does this mean and the difference?"

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
    system_prompt = """You are an assistant that generates alternative queries for a RAG system focused on Paul Graham's essays. 
Generate 3-5 alternative queries that:
1. Explore different angles of the original question
2. Use related concepts Paul Graham discusses
3. Rephrase using synonyms while maintaining intent
4. Uncover broader or deeper context

Return only a JSON array of the alternative queries in the format:
```json
{
    "alternative_queries": [
        "Alternative query 1",
        "Alternative query 2",
        "Alternative query 3",
        ...
    ]
}
```"""
    
    user_prompt = f"""Original query: "{query}"

Generate alternative queries for this question about Paul Graham's work."""
    
    response = model.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={'type': 'json_object'},
        temperature=0.7,
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        print("Response from the model:", result)
        # Expect either {"queries": [...]} or {"alternative_queries": [...]}
        queries = result.get("queries", result.get("alternative_queries", []))
        return [query] + queries  # Include original
    except (json.JSONDecodeError, KeyError):
        print("Failed to parse response, using original query only")
        return [query]

queries = generate_multi_query(query)
print("Generated Queries:", queries)

# flatten the queries if they are nested lists
joint_query = [item for sublist in queries for item in sublist] if isinstance(queries[0], list) else queries

print("\nJoint Query:", joint_query)

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

# Step 2: Deduplicate documents + metadata
unique_pairs = list({(doc, meta["filename"]): (doc, meta) 
                     for doc, meta in zip(retrieved_documents[0], retrieved_metadatas[0])}.values())
unique_documents = [doc for doc, meta in unique_pairs]

# Step 3: Score documents with Cross-Encoder
pairs = [[query, doc] for doc in unique_documents]
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
    system_prompt = """You are an AI assistant that answers questions using Paul Graham's essays. 

Rules:
1. Use ONLY the provided context - don't add external information
2. Prioritize the most relevant excerpts (typically the first 3-5)
3. If the context doesn't fully answer the question, be honest about limitations
4. When quoting, use exact text from the context

Response format: JSON with "answer", "key_quote", "source", and "confidence" fields."""
    
    user_prompt = f"""Context from Paul Graham's essays:
{context}

Question: {query}

Based on this context, provide a comprehensive answer."""
    
    response = model.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={'type': 'json_object'},
        temperature=0.2,  # Low temperature for factual accuracy
        max_tokens=800,   # Reasonable limit
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        
        # Ensure required fields exist
        return {
            "answer": result.get("answer", "Unable to generate answer from context."),
            "key_quote": result.get("key_quote", result.get("quote", "")),
            "confidence": result.get("confidence", "low"),
            "source": result.get("source", "Paul Graham's essays"),
            "query": query  # Preserve original query for debugging
        }
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return {
            "answer": "Failed to parse model response.",
            "key_quote": "",
            "confidence": "low",
            "source": "Paul Graham's essays",
            "query": query
        }

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