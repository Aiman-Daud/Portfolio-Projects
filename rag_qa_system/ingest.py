"""Run once after adding files to data/: builds the vector index."""
from rag import build_index

if __name__ == "__main__":
    build_index()
