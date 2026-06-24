from src.retrieval.vector_store import VectorStore
vs = VectorStore()
points, payloads = vs.get_all_vectors()
shapes = set(len(p) for p in points)
print("Shapes in points:", shapes)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
qv = emb.embed_query("test")
print("Query vector len:", len(qv))
try:
    import numpy as np
    all_vectors = points + [qv]
    np.array(all_vectors)
    print("NumPy array created successfully!")
except Exception as e:
    print("NumPy error:", e)
