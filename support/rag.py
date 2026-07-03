import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from django.conf import settings
from google import genai
import os
from typing import Any, Dict
from pypdf import PdfReader


gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function backed by Google's free Gemini embedding API.
    Replaces chromadb's DefaultEmbeddingFunction (which needs onnxruntime,
    the source of the Windows DLL failure) with a hosted, dependency-free call."""

    def __init__(self, task_type: str = "RETRIEVAL_DOCUMENT"):
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=list(input),
            config={"task_type": self.task_type},
        )
        return [e.values for e in result.embeddings]

    @staticmethod
    def name() -> str:
        return "gemini-embedding-001"

    def get_config(self) -> Dict[str, Any]:
        return {"model": "gemini-embedding-001", "task_type": self.task_type}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(task_type=config.get("task_type", "RETRIEVAL_DOCUMENT"))


# initialize chromadb client
client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

embedding_fn = GeminiEmbeddingFunction()  # RETRIEVAL_DOCUMENT, used for indexing

# get or create collection - just like table in regular db
collection = client.get_or_create_collection(
    name="coolbreeze_docs",
    embedding_function=embedding_fn
)

# separate embedding function for queries — same model, different task_type
query_embedding_fn = GeminiEmbeddingFunction(task_type="RETRIEVAL_QUERY")


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1

        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def load_documents():
    docs_path = "support/documents/"

    documents = []
    ids = []

    for filename in os.listdir(docs_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_path, filename)
            reader = PdfReader(filepath)

            raw_text = ""
            for page in reader.pages:
                raw_text += page.extract_text()

            chunks = chunk_text(raw_text, chunk_size=500)

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{filename}_{i}")

    if documents:
        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            print(f"Adding batch {i // batch_size + 1} ({len(batch_docs)} chunks)...")
            collection.add(documents=batch_docs, ids=batch_ids)

    print(f"Loaded {len(documents)} chunks into ChromaDB")


def search_knowledge_base(query):
    try:
        query_embedding = query_embedding_fn([query])[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        print("DEBUG RESULTS:", results["documents"])
        if not results["documents"][0]:
            return "No relevant information found in company documents."

        matched_chunks = results["documents"][0]
        return "\n\n".join(matched_chunks)
    except Exception as e:
        print("RAG search failed:", e)
        return "Company document search is temporarily unavailable. Please answer using general knowledge or ask the customer to contact support directly for policy details."