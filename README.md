# Agentic RAG Document Analysis System

![Agentic RAG](https://img.shields.io/badge/Architecture-LangGraph-blue)
![Python](https://img.shields.io/badge/Python-3.11+-yellow)
![Database](https://img.shields.io/badge/Vector_DB-Qdrant-red)

A highly robust, multi-agent Retrieval-Augmented Generation (RAG) system built with LangGraph, Streamlit, and Qdrant. This project is specifically designed to parse, synthesize, and rigorously critique responses derived from engineering documents, ensuring zero hallucinations.

## 🌟 Key Features
- **Multi-Agent Architecture**: Employs a specific `Researcher -> Synthesizer -> Critic` workflow. The Critic agent actively checks answers for hallucinations and forces rewrites if necessary.
- **Glassmorphic Visualizer UI**: A beautiful, custom SVG/CSS animated graphical interface that streams the real-time state of the agents directly in the Streamlit browser.
- **Robust Edge Case Handling**: Catching all potential points of failure, including empty database results, corrupted documents, and malformed LLM JSON parsing.
- **HyDE (Hypothetical Document Embeddings)**: Enhances retrieval accuracy by generating hypothetical answers to index against before doing keyword search.
- **Zero-Setup Local Database**: Uses Qdrant Local mode to seamlessly index uploaded PDF documents right out of the box.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/AgenticProject.git
   cd AgenticProject
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and run Ollama**:
   This project defaults to running Llama3 locally. You must have [Ollama](https://ollama.com/) installed and running.
   ```bash
   ollama serve
   ollama run llama3
   ```

## 🚀 Usage

Start the Streamlit application:
```bash
streamlit run app.py
```
1. Open the sidebar and **upload** your engineering PDFs.
2. Enter your query in the chat box.
3. Watch the **Interactive Visualizer** light up as the `Researcher`, `Synthesizer`, and `Critic` agents route the information and form the best answer.

## 🧪 Running Tests
The project contains a comprehensive suite of unit tests. Run them using pytest:
```bash
python -m pytest tests/ -v --cov=src
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
