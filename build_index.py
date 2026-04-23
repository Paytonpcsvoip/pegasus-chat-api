import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

WP_BASE_URL = "https://pegasuscs.com/wp-json/wp/v2/"
INDEX_PATH = "./wp_semantic_index"

def clean_content(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())

def fetch_all(endpoint):
    items = []
    page = 1
    while True:
        url = f"{WP_BASE_URL}{endpoint}?per_page=100&page={page}"
        resp = requests.get(url)
        if resp.status_code != 200:
            break
        data = resp.json()
        if not isinstance(data, list) or not data:
            break
        items.extend(data)
        page += 1
    return items

print("Fetching content from pegasuscs.com...")
all_items = fetch_all("pages") + fetch_all("posts")
print(f"Total items fetched: {len(all_items)}")

documents = []
for item in all_items:
    title = item.get("title", {}).get("rendered", "")
    content = clean_content(item.get("content", {}).get("rendered", ""))
    url = item.get("link", "")
    if len(content) < 50:
        continue
    documents.append(Document(
        page_content=content,
        metadata={"title": title, "source_url": url}
    ))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

print("Building semantic index with Grok embeddings...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ.get("OPENAI_API_KEY")   # Temporary - we'll switch fully later if needed
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=INDEX_PATH
)

print("✅ Index built successfully!")
print(f"Total chunks: {len(chunks)}")