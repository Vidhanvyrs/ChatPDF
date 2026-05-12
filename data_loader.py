#use my index to load in PDF documents and to embed them (we need to create the vectors)
import os
from huggingface_hub import InferenceClient
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Using HuggingFace Inference API (Free & Light for Vercel)
hf_token = os.getenv("HUGGINGFACE_TOKEN")
client = InferenceClient(token=hf_token)

# The model you used before, but now running as a free API
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384 

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # Call the free HuggingFace Inference API
    embeddings = client.feature_extraction(texts, model=EMBED_MODEL)
    # The return is a nested list or numpy-like array
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return list(embeddings)