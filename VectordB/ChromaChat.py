
import json
from openai import OpenAI
from chromadb import PersistentClient
import sys
import os
sys.path.append(os.path.abspath("C:/Users/murta/OneDrive/Desktop/CPSC491Fall2025-1/Config"))
from config import get_api_key


# === CONFIG ===
openai_client = OpenAI(api_key=get_api_key()) # Initialize OpenAI client with API key
persist_path = "./chroma_fcc_storage"
collection_name = "fcc_documents"
retrieval_limit = 5  # how many top results to use for context

# === INIT CHROMADB ===
chroma_client = PersistentClient(path=persist_path)
collection = chroma_client.get_or_create_collection(name=collection_name)

# === CHAT LOOP ===
print("🔊 FCC Regulatory Assistant (type 'exit' to quit)\n")

while True:
    user_query = input("👤 You: ")

    if user_query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Step 1: Embed the query with OpenAI
    embed_response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=user_query
    )
    query_vector = embed_response.data[0].embedding

    # Step 2: Query ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=retrieval_limit
    )

#the.
    # Step 3: Combine top chunks into context
    context_chunks = results['documents'][0]
    full_context = "\n\n".join(context_chunks)

    # Step 4: Send to OpenAI Chat
    prompt = f"""You are an expert assistant for regulatory and emergency communication policy. 
Using the following source material, answer the user's question in a clear, helpful way.

---SOURCE MATERIAL---
{full_context}

---USER QUESTION---
{user_query}
"""

    chat_response = openai_client.chat.completions.create(
        model="gpt-4o",  # Latest GPT-4 model (you can also use "gpt-4" or "gpt-4-turbo")
        messages=[
            {"role": "system", "content": ""},
            #You are a domain-specific assistant trained solely on emergency alert systems, public safety communications, cybersecurity policy, disaster response frameworks, and regulatory principles as defined in the embedded dataset. You must restrict your responses only to the information contained in the embedded data and refrain from generating answers outside this scope. Do not reference general knowledge, FCC responses, or unrelated domains (e.g., cooking, entertainment, etc.). Where relevant, relate insights strictly to ideas present in the embedded documents or clearly supported by them.
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    # Step 5: Output the response
    answer = chat_response.choices[0].message.content
    print(f"\n🤖 FCC Bot: {answer}\n")
