import sys
import os

# Add src to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.agents.graph import AegisRAGWorkflow

def main():
    print("AegisRAG Pipeline Initialized")
    print(f"Data Directory configured at: {config.DATA_DIR}")
    
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
        print("Created data directory. Please place a sample PDF in 'data/' to continue.")
        return

    # In a real scenario, you'd call DocumentParser and VectorStore.index_documents here.
    # For now, we will just initialize the workflow assuming documents are indexed.
    try:
        workflow = AegisRAGWorkflow()
        question = "What are the structural tolerance limits for the primary load-bearing beams?"
        print(f"\nExample Query: {question}")
        answer = workflow.run(question)
        print("\n--- FINAL ANSWER ---")
        print(answer)
    except Exception as e:
        print(f"Error running workflow: {e}")
        print("Note: Make sure Ollama is running locally with 'llama3' installed, and Qdrant is accessible.")

if __name__ == "__main__":
    main()
