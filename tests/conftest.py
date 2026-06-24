import pytest
from src.config import config
import os

@pytest.fixture(autouse=True)
def set_test_env():
    # Force memory database for tests
    config.QDRANT_URL = ":memory:"
    config.DATA_DIR = "/tmp/test_data"
