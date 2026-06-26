import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    
    # Default to local docker server if no URL is provided
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_PATH: str = Field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "..", "qdrant_db"))
    COLLECTION_NAME: str = Field(default="dense_docs")
    DATA_DIR: str = Field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "..", "data"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

config = Config()
