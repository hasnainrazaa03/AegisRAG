import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    # Default to local persistent storage if no URL is provided
    QDRANT_URL = os.getenv("QDRANT_URL", None)
    QDRANT_PATH = os.path.join(os.path.dirname(__file__), "..", "qdrant_db")
    COLLECTION_NAME = "dense_docs"
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

config = Config()
