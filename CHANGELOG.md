# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-24

### Added
- **Multi-Agent RAG Pipeline**: Implemented `ResearcherAgent`, `SynthesizerAgent`, and `CriticAgent` using LangGraph.
- **Glassmorphic Interactive Visualizer**: SVG and CSS based animated progress UI injected natively into Streamlit to visualize agent states in real-time.
- **Dynamic PDF Uploading**: Users can upload `.pdf` engineering documents via the sidebar which are chunked and indexed into the database on the fly.
- **HyDE Retrival Optimization**: Hypothetical Document Embeddings for superior context matching.
- **Comprehensive Unit Testing**: A robust `pytest` suite simulating edge cases, LLM failures, and mocking the database.
- **GitHub Actions Pipeline**: `.github/workflows/test.yml` running Pytest and Codecov on every commit.

### Fixed
- Re-architected Qdrant local VectorStore instantiation as a Streamlit singleton (`@st.cache_resource`) to prevent process locking crashes (`Errno 35 Resource temporarily unavailable`).
- `CriticAgent` now gracefully catches malformed JSON output from LLMs and correctly routes to "rewrite".
- `SynthesizerAgent` now gracefully returns a generic message if Context is empty.
- Avoided `token_type_ids` bug in ONNX Runtime by switching the local FlashRank model to `ms-marco-TinyBERT-L-2-v2`.
