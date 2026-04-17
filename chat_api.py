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

# Grok embeddings + Grok answers (no local Ollama)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ.get("GROK_API_KEY"),
    openai_api_base="https://api.x.ai/v1"
)

vectorstore = Chroma(
    persist_directory="./wp_semantic_index",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.environ.get("GROK_API_KEY")
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
   context = "You are answering general PCS VoIP questions without a knowledge base."
sources = []

        system_prompt = """You are a helpful assistant for Pegasus Communication Solutions.
Answer using only the provided context. Be professional, clear, and friendly."""

        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.message}"}
            ],
            temperature=0.7,
            max_tokens=800
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources[:5]
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "answer": "Sorry, I'm having trouble right now. Please try again in a moment.",
            "sources": []
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))