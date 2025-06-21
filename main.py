import getpass
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

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

# response = model.invoke("Are you there?")
# print(response)



query = "How do I tackle imposter syndrome in my career?"

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pg_essays")

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
    
    print("Response from the model:", response)

    try:
        queries = json.loads(response.choices[0].message.content)["queries"]
    except json.JSONDecodeError:
        print("Failed to parse JSON response from the model.")
        queries = []
    return queries

aug_queries = generate_multi_query(query)

# 1. First step show the augmented queries
# for query in aug_queries:
#     print("\n", query)
