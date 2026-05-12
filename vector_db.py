from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

#turning docs into vectors and storing them for similarity searches
class QdrantStorage:
    def __init__(self, url=None, collection="docs", dim=384):
        import os
        qdrant_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key,
            timeout=30
        )
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

# search vector database and get the relevant results based on this query vector 
    def search(self, query_vector, top_k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points

        contexts = []
        sources = set()

        for r in results:
            payload = r.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}