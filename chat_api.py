from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
import uvicorn
import os
from openai import OpenAI

app = FastAPI(title="Pegasus CS Assistant")

# CORS - allows your WordPress site to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use Grok's embedding model
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

# Grok for generating answers
client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.environ.get("GROK_API_KEY")
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Get relevant content from your semantic index
        docs = retriever.invoke(request.message)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get("source_url") for doc in docs if doc.metadata.get("source_url")]

        # Generate natural answer using Grok
        system_prompt = """You are a helpful, professional assistant for Pegasus Communication Solutions (PCS VoIP). 
Use only the provided context to answer the user's question.
Be clear, friendly, and accurate. If the information is not in the context, politely say so."""

        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context from our knowledge base:\n{context}\n\nUser Question: {request.message}"}
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
            "answer": "Sorry, I'm having trouble connecting right now. Please try again in a moment.",
            "sources": []
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))