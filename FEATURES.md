# Features Tracker

This document tracks upcoming features, structural architectural changes, and known issues.

## Planned Features

- [ ] **Inline Source Citations & References**: Synthesizer Agent explicitly cites the documents/web pages it used, displaying clickable footnotes in the UI.
- [x] **Streaming Answers**: Stream text token-by-token into the Streamlit UI to make it feel blazing fast.
- [x] **Active Document Grading (Self-RAG)**: A new `GraderAgent` to discard irrelevant retrieved documents before the Synthesizer sees them.
- [x] **Export to PDF/Report**: Add a button to export the finalized answer and references as a formatted PDF or Markdown report.
- [x] **Human-in-the-Loop (HIL)**: Pause execution before the Synthesizer runs to let the user review and accept/reject retrieved context.

## Completed Features
- [x] **Multi-Model Router**: Add a dropdown in the UI to dynamically switch between local `llama3` and cloud-hosted `claude-3-5-sonnet`.
- [x] **Conversational Memory**: Update the `AegisRAGState` to store full conversation history so agents can perform multi-turn context lookups.
- [x] **Web Search Tool**: Give the `ResearcherAgent` a Tavily/DuckDuckGo tool to fetch real-time web results if the internal database yields empty context.
- [x] **Supervisor Agent**: A master agent that decides *which* sub-agents need to run (e.g. bypassing the Synthesizer if a direct exact match was found).
- [x] **Streaming Answers**: `SynthesizerAgent` now streams output token-by-token to the Streamlit UI for near-instant perceived generation.
- [x] **Active Document Grading (Self-RAG)**: `GraderAgent` evaluates relevance of retrieved context chunks and explicitly filters out bad documents to prevent hallucination context bloat.

## Known Issues

- Qdrant Local locks the database directory to a single thread. Restarting the Streamlit application rapidly without gracefully killing the background process can result in `AlreadyLocked: [Errno 35]` errors. (Current workaround: `pkill -f streamlit`).
- Streamlit's `unsafe_allow_html` is required for the SVG agent visualizer. Upgrading Streamlit in the future may break or sandbox these injections.
