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