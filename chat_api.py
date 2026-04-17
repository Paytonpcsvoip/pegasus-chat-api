from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import uvicorn
import os
from openai import OpenAI

app = FastAPI(title="Pegasus CS Assistant")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embeddings + Vector Store
try:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    vectorstore = Chroma(
        persist_directory="./wp_semantic_index",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    print("✅ Vector store loaded successfully")
except Exception as e:
    print(f"❌ Failed to load vector store: {e}")

# Grok Client
client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.environ.get("GROK_API_KEY")
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Retrieve context
        docs = retriever.invoke(request.message)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get("source_url") for doc in docs if doc.metadata.get("source_url")]

        # Generate answer with Grok
        system_prompt = """You are a helpful assistant for Pegasus Communication Solutions.
Use only the provided context to answer the question accurately and professionally.
If the answer is not in the context, say "I don't have that specific information in our current knowledge base.""""

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
        print(f"Error processing request: {e}")
        return {
            "answer": "Sorry, I'm having trouble processing your request right now. Please try again in a moment.",
            "sources": []
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))