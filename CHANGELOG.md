# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Inline Source Citations & References**: `SynthesizerAgent` now properly injects references into its answers (`[1]`, `[2]`), and `app.py` renders an expandable 'References & Sources' list at the bottom of the message.
- **Query Contextualization**: `ResearcherAgent` reformulates queries referencing chat history to stand-alone queries prior to searching vector store or web.
- **Multi-Model Support**: Dropdown added in sidebar to hot-swap between `llama3` and `claude-3-5-sonnet`.
- **Streaming Answers**: `SynthesizerAgent` now streams output token-by-token to the Streamlit UI.
- **Active Document Grading (Self-RAG)**: `GraderAgent` evaluates relevance of retrieved context chunks and explicitly filters out bad documents to prevent hallucination context bloat.
- **Export to Report**: Users can now download a clean Markdown `.md` report containing their question, the synthesized answer, and references directly from the chat UI.
- **Human-in-the-Loop (HIL)**: Graph execution now natively pauses (`interrupt_before`) prior to the synthesis stage, allowing users to manually approve or reject the graded context via the UI.

## [1.0.0] - 2026-06-24

### Added
- **Multi-Model Support**: Added a UI dropdown to dynamically hot-swap between local `llama3` and cloud-hosted `claude-3-5-sonnet`.
- **Supervisor Agent**: Implemented `SupervisorAgent` at the start of the workflow to route generic questions directly to answers without going through the heavy RAG pipelines.
- **Web Search Fallback**: Added `DuckDuckGoSearchRun` to the `ResearcherAgent` to dynamically fetch answers from the internet if the local vector store yields no relevant context.
- **Conversational Memory**: Updated `AegisRAGState`, `AegisRAGWorkflow`, `SynthesizerAgent`, and Streamlit UI to maintain and utilize multi-turn chat history.
- **Multi-Agent RAG Pipeline**: Implemented `ResearcherAgent`, `SynthesizerAgent`, and `CriticAgent` using LangGraph.
- **Glassmorphic Interactive Visualizer**: SVG and CSS based animated progress UI injected natively into Streamlit to visualize agent states in real-time.
- **Dynamic PDF Uploading**: Users can upload `.pdf` engineering documents via the sidebar which are chunked and indexed into the database on the fly.
- **HyDE Retrival Optimization**: Hypothetical Document Embeddings for superior context matching.
- **Comprehensive Unit Testing**: A robust `pytest` suite simulating edge cases, LLM failures, and mocking the database.
- **GitHub Actions Pipeline**: `.github/workflows/test.yml` running Pytest and Codecov on every commit.

### Changed
- **Visualizer Redesign**: Completely overhauled the Streamlit SVG visualizer to use a 3-node connected layout with MagicUI-style animated circular progress rings, dynamic numerical progress counters, and modernized typography.

### Fixed
- Re-architected Qdrant local VectorStore instantiation as a Streamlit singleton (`@st.cache_resource`) to prevent process locking crashes (`Errno 35 Resource temporarily unavailable`).
- `CriticAgent` now gracefully catches malformed JSON output from LLMs and correctly routes to "rewrite".
- `SynthesizerAgent` now gracefully returns a generic message if Context is empty.
- Avoided `token_type_ids` bug in ONNX Runtime by switching the local FlashRank model to `ms-marco-TinyBERT-L-2-v2`.
