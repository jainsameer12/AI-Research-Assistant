from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# 🔹 Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 Global storage
documents_store = []
index = None


# 🔹 Build Vector Store
def build_vector_store(chunks):
    global index, documents_store

    documents_store = chunks

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))


# 🔹 Retrieve relevant chunks
def retrieve_context(query):
    global index, documents_store

    query_embedding = model.encode([query])

    # 🔥 reduced k for better focus
    distances, indices = index.search(query_embedding, k=2)

    results = []

    for i in indices[0]:
        results.append({
            "text": documents_store[i]
        })

    return results


# 🔹 Generate Answer with Chat Memory + Strict Control
def generate_answer(query, chat_history=None):
    global index

    # ❗ If no document uploaded
    if index is None:
        return "Please upload and process a document first.", []

    if chat_history is None:
        chat_history = []

    results = retrieve_context(query)

    context = "\n\n".join([r["text"] for r in results])

    # 🔹 Build conversation history
    history_text = ""
    for q, a in chat_history:
        history_text += f"Q: {q}\nA: {a}\n"

    # 🔥 Strong prompt to avoid hallucination
    prompt = f"""
You are an AI assistant.

STRICT RULES:
- Answer ONLY using the provided context
- Format answer in bullet points if listing items
- Keep answer short and clear
- Do NOT add extra information

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",   # ✅ using better model
                "prompt": prompt,
                "stream": False
            }
        )

        res_json = response.json()

        # 🔹 Safe response handling
        if "response" in res_json:
            answer = res_json["response"]
        else:
            answer = f"Error from model: {res_json}"

    except Exception as e:
        answer = f"Error: {str(e)}"

    return answer, results