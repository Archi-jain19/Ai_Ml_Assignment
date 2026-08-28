import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import build_retrieval_index, load_enriched_facets

def main():
    print("Rebuilding FAISS retrieval index from enriched facets...")
    df = load_enriched_facets()
    build_retrieval_index(df)
    print("FAISS retrieval index successfully rebuilt!")

if __name__ == "__main__":
    main()
