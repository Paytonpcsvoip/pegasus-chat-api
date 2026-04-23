from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import uvicorn
import os
from openai import OpenAI

app = FastAPI(title="Pegasus CS Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Grok Embeddings (learns your 80 pages)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ.get("GROK_API_KEY"),
    openai_api_base="https://api.x.ai/v1"
)

vectorstore = Chroma(
    persist_directory="./wp_semantic_index",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 7})  # Get 7 best matches

client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.environ.get("GROK_API_KEY")
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        docs = retriever.invoke(request.message)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get("source_url") for doc in docs if doc.metadata.get("source_url")]

        system_prompt = """You are a helpful assistant for Pegasus Communication Solutions.
You help users with PCS VoIP, automation manager, mobile app, billing, users, etc.
Answer using ONLY the provided context from the website. 
Be professional, clear, and friendly. If you don't know, say so."""

        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context from our help site:\n{context}\n\nUser Question: {request.message}"}
            ],
            temperature=0.7,
            max_tokens=900
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources[:6]
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "answer": "Sorry, I'm having trouble connecting right now. Please try again.",
            "sources": []
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))