# Features Tracker

This document tracks upcoming features, structural architectural changes, and known issues.

## Planned Features

- [ ] **Multi-Model Router**: Add a dropdown in the UI to dynamically switch between local `llama3` and cloud-hosted `claude-3-5-sonnet`.
- [ ] **Conversational Memory**: Update the `AegisRAGState` to store full conversation history so agents can perform multi-turn context lookups.
- [ ] **Web Search Tool**: Give the `ResearcherAgent` a Tavily/DuckDuckGo tool to fetch real-time web results if the internal database yields empty context.
- [ ] **Supervisor Agent**: A master agent that decides *which* sub-agents need to run (e.g. bypassing the Synthesizer if a direct exact match was found).

## Known Issues

- Qdrant Local locks the database directory to a single thread. Restarting the Streamlit application rapidly without gracefully killing the background process can result in `AlreadyLocked: [Errno 35]` errors. (Current workaround: `pkill -f streamlit`).
- Streamlit's `unsafe_allow_html` is required for the SVG agent visualizer. Upgrading Streamlit in the future may break or sandbox these injections.
