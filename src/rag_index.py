import os
import time
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = 'outputs/chroma_store'
COLLECTION_NAME = 'mrna_safety_guidelines'

def load_and_chunk_document(doc_path: str, chunk_size: int=400, chunk_overlap: int=50) -> List[str]:
    """
    Load the safety guideline reference text and split it into overlapping chunks (FR-3.1).

    Args:
        doc_path (str): Path to the guidelines text file.
        chunk_size (int): Target character/token chunk length (~300-500 tokens).
        chunk_overlap (int): Overlap between consecutive chunks (~50 tokens).
    
    Returns:
        List[str]: List of text chunk strings.
    """
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Guideline reference document not found at: {doc_path}")

    with open(doc_path, 'r', encoding='UTF-8') as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '. ', ' ']
    )
    chunks = splitter.split_text(raw_text)
    print(f"Successfully loaded '{doc_path}' and split into {len(chunks)} chunks.")
    return chunks

def build_or_load_vectorstore(doc_path: str=None, force_rebuild:bool=False) -> Chroma:
    """
    Build vector embeddings and persist them in ChromaDB, or load existing store (FR-3.2, FR-3.3).

    Args:
        doc_path (str): Path to guidelines document. Required if building for the first time.
        force_rebuild (bool): If True, re-indexes source document even if local database exists.
    
    Returns:
        Chroma: Initialized local Chroma vector database instance.
    """
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Load existing database if available and rebuild not forced
    if os.path.exists(PERSIST_DIR) and not force_rebuild:
        print(f"Loading existing persisted vector store from '{PERSIST_DIR}'...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )
        return vectorstore

    if not doc_path:
        raise ValueError("doc_path must be provided to build a new vector index.")

    # Create new index from chunks
    chunks = load_and_chunk_document(doc_path)
    print("Generating dense vector embeddings and populating ChromaDB store...")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    # Persist database to disk
    vectorstore.persist()
    print(f"Vector store successfully persisted to '{PERSIST_DIR}'")
    return vectorstore

def query_guidelines(vectorstore: Chroma, question: str, k: int=3) -> Tuple[List[str], float]:
    """
    Perform semantic similarity query against indexed guidelines and log latency (FR-3.4, FR-3.5).

    Args:
        vectorstore (Chroma): The loaded vector store.
        question (str): Natural Language question.
        k (int): Number of top semantic matches to retrieve.
    
    Returns:
        Tuple[List[str], float]: Retrieved chunk strings and query execution time in seconds.
    """
    start_time = time.time()
    results = vectorstore.similarity_search(question, k=k)
    elapsed_time = time.time() - start_time

    retrieved_chunks = [doc.page_content for doc in results]
    print(f"\nQuery: '{question}'")
    print(f"Retrieved {len(retrieved_chunks)} relevant chunk(s) in {elapsed_time:.4f} seconds.")

    return retrieved_chunks, elapsed_time

if __name__ == "__main__":
    guidelines_path = "data/mrna_safety_guidelines.txt"

    # Step 1: Build or load the vector database (FR-3.1, FR-3.2, FR-3.3)
    store = build_or_load_vectorstore(doc_path=guidelines_path)

    # Step 2: Verification with candidate questions (FR-3.4, FR-3.5)
    text_questions = [
        "What are the storage requirements below -20C?",
        "What is the maximum safe dose for patients requiring secondary clinician review?",
        "What are the age-banded dosing rules for pediatric patients under 5 years?",
        "What are the absolute contraindications related to lipid nanoparticle allergic reactions?",
        "How within how many hours must moderate or severe adverse events be documented?"
    ]

    print("\nRAG Retrieval Verification".center(20, "="))

    for q in text_questions:
        chunks, latency = query_guidelines(store, question=q, k=3)
        assert len(chunks) == 3, "Expected exactly k=3 chunks per query"
        assert 0.0 < latency < 5.0, f"Query latency {latency:.3f}s outside expected threshold (<5s)"
        print(f"Top Result Snippet:\n{chunks[0][:150]}...\n" + "-"*50)