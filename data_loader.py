#use my index to load in PDF documents and to embed them (we need to create the vectors)
from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Using a free local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#pdf can be large so we need to chunk it into smaller pieces
#llamaindex will read the pdf and chunk it out
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # This model produces 384-dimensional embeddings

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

#chunking process we are taking pdf and turning it into smaller pieces and then embedding it 
def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # Using local sentence transformer model - no API calls needed!
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
    