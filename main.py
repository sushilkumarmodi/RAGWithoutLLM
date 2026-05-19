from sentence_transformers import SentenceTransformer  # type: ignore[import]
import faiss  # type: ignore[import]
import numpy as np  # type: ignore[import]

model = SentenceTransformer('all-MiniLM-L6-v2')

docs = [
    "SQL is a structured query language used to manage relational databases",
    "SQL is used for querying and updating data",
    "Python is used for data science and machine learning",
    "Power BI is used to build dashboards and reports"
]

# ✅ embeddings
embeddings = model.encode(docs)
embeddings = np.array(embeddings, dtype='float32')

# ✅ normalize
faiss.normalize_L2(embeddings)

# ✅ cosine similarity index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# ✅ query
query = "What is SQL?"
q_embedding = model.encode([query])
q_embedding = np.array(q_embedding, dtype='float32')

faiss.normalize_L2(q_embedding)

# ✅ search
D, I = index.search(q_embedding, 1)

# ✅ output
for i in I[0]:
    print("✅ Best Answer:")
    print(docs[i])
