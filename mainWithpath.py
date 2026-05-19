import faiss # type: ignore[import]
from sentence_transformers import SentenceTransformer # type: ignore[import]
import os
import numpy as np # type: ignore[import]

# ✅ File readers
from PyPDF2 import PdfReader # type: ignore[import]
from docx import Document # type: ignore[import]

# ✅ Load embedding model (NOT LLM)
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# ✅ FILE READERS
# -----------------------------

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_pdf(file_path):
    text = ""
    pdf = PdfReader(file_path)
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text


def read_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def read_json(file_path):
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return str(data)


# -----------------------------
# ✅ FILE LOADER (AUTO-DETECT)
# -----------------------------

def load_document(file_path):
    ext = file_path.split('.')[-1].lower()

    if ext == "txt":
        return read_txt(file_path)
    elif ext == "pdf":
        return read_pdf(file_path)
    elif ext == "docx":
        return read_docx(file_path)
    elif ext == "json":
        return read_json(file_path)
    else:
        print(f"Unsupported file: {file_path}")
        return ""


# -----------------------------
# ✅ TEXT CHUNKING (IMPORTANT)
# -----------------------------

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# ✅ LOAD ALL FILES FROM FOLDER
# -----------------------------

def load_all_documents(folder_path):
    all_chunks = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        print(f"Reading: {file_name}")

        text = load_document(file_path)
        chunks = chunk_text(text)

        all_chunks.extend(chunks)

    return all_chunks


# -----------------------------
# ✅ BUILD VECTOR DATABASE
# -----------------------------


def create_vector_db(chunks):

    # ✅ Step 1: Check empty input
    if len(chunks) == 0:
        raise ValueError("Chunks list is empty")

    # ✅ Step 2: Generate embeddings
    embeddings = model.encode(chunks)

    # ✅ Step 3: Convert to NumPy float32
    embeddings = np.array(embeddings, dtype="float32")

    # ✅ Step 4: Ensure 2D shape
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be 2D array")

    # ✅ Step 5: Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # ✅ Step 6: Add embeddings
    index.add(embeddings)

    return index, chunks



# -----------------------------
# ✅ SEARCH FUNCTION
# -----------------------------

def search(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])

    D, I = index.search(query_embedding, top_k)

    print("\n✅ Results:\n")

    for idx in I[0]:
        print(chunks[idx])
        print("\n---\n")


# -----------------------------
# ✅ MAIN FUNCTION
# -----------------------------

def main():

    folder_path = "/workspaces/RAGWithoutLLM/documents"   # 👈 Put files here: pdf/docx/txt/json

    print("Loading documents...")
    chunks = load_all_documents(folder_path)

    print("Creating vector DB...")
    index, chunks = create_vector_db(chunks)

    print("\n✅ System Ready!\n")

    while True:
        query = input("Enter your question (or 'exit'): ")

        if query.lower() == "exit":
            break

        search(query, index, chunks)


if __name__ == "__main__":
    main()