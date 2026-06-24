import streamlit as st
import os
import sys
import tempfile

# Add src to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.agents.graph import AegisRAGWorkflow
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.visualizer import get_agent_graph_html

st.set_page_config(page_title="AegisRAG for Engineering", layout="wide", page_icon="🛡️")

st.title("🛡️ AegisRAG Pipeline")
st.markdown("""
This application uses a multi-agent workflow to retrieve and synthesize answers from dense engineering manuals.
Watch the agents process your query in real-time!
""")

# ---- SIDEBAR CONFIGURATION ----
st.sidebar.header("Agent Configuration")

model_choice = st.sidebar.selectbox(
    "Choose LLM Engine:",
    ("Local (Ollama Llama 3)", "Cloud (Claude 3.5 Sonnet)")
)

max_iterations = st.sidebar.slider("Max Critic Iterations", min_value=1, max_value=5, value=3)
use_hyde = st.sidebar.checkbox("Enable HyDE (Hypothetical Document Embeddings)", value=True)

# Initialize LLM based on user selection
def get_llm():
    if model_choice == "Cloud (Claude 3.5 Sonnet)":
        if not config.ANTHROPIC_API_KEY:
            st.sidebar.error("⚠️ ANTHROPIC_API_KEY not found in .env")
            return None
        return ChatAnthropic(model_name="claude-3-5-sonnet-20240620", anthropic_api_key=config.ANTHROPIC_API_KEY)
    else:
        return ChatOllama(model="llama3")

llm = get_llm()

from src.retrieval.vector_store import VectorStore

# Initialize VectorStore globally so it doesn't try to open Qdrant multiple times
@st.cache_resource
def get_vector_store():
    return VectorStore()

vector_store = get_vector_store()

# Initialize Workflow with settings globally so we can use its VectorStore for uploads
@st.cache_resource(hash_funcs={ChatOllama: id, ChatAnthropic: id})
def get_workflow(_llm, _use_hyde, _max_iterations):
    if _llm is None:
        return None
    return AegisRAGWorkflow(llm=_llm, use_hyde=_use_hyde, max_iterations=_max_iterations, vector_store=vector_store)

workflow = get_workflow(llm, use_hyde, max_iterations)

st.sidebar.divider()
st.sidebar.header("📚 Document Uploader")
uploaded_file = st.sidebar.file_uploader("Upload an Engineering Manual (.pdf, .txt)", type=["pdf", "txt"])

if uploaded_file and st.sidebar.button("Ingest Document"):
    if workflow is None:
        st.sidebar.error("Workflow not loaded.")
    else:
        with st.sidebar.status("Ingesting document..."):
            try:
                # Save uploaded file temporarily to process it
                file_ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Load the file
                st.write("Loading file...")
                if file_ext.lower() == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                
                docs = loader.load()
                
                # Also save permanently to data directory for future batch runs
                perm_path = os.path.join(config.DATA_DIR, uploaded_file.name)
                with open(perm_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.write("Splitting text into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)
                
                st.write("Indexing into Qdrant database...")
                # We use the already-open VectorStore instance bound to the Streamlit session
                vs = workflow.researcher.retriever.vector_store
                vs.index_documents(chunks)
                
                st.toast(f"✅ Successfully ingested {uploaded_file.name}!")
                st.write("Done!")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

st.sidebar.divider()
st.sidebar.header("System Status")
st.sidebar.write(f"**Data Directory:** `{config.DATA_DIR}`")
st.sidebar.write("**Vector Store:** `Qdrant (Local)`")
st.sidebar.write("**Re-ranker:** `FlashRank (TinyBERT)`")

# ---- MAIN INTERFACE ----

query = st.chat_input("Enter your engineering query (e.g., structural tolerance limits)")

if query:
    if workflow is None:
        st.error("Please configure your API keys to use the selected model.")
        st.stop()
        
    # Display User Query
    with st.chat_message("user"):
        st.write(query)
        
    # Stream execution using the Custom HTML/SVG Visualizer
    with st.chat_message("assistant"):
        visualizer_container = st.empty()
        
        try:
            # Initial render: Researcher active
            visualizer_container.markdown(get_agent_graph_html("research"), unsafe_allow_html=True)
            
            final_draft = ""
            for s in workflow.stream(query):
                # 's' is a dictionary keyed by the node name
                for node, state in s.items():
                    if node == "research":
                        # Transition to Synthesizer
                        visualizer_container.markdown(get_agent_graph_html("synthesize"), unsafe_allow_html=True)
                    elif node == "synthesize":
                        # Transition to Critic
                        visualizer_container.markdown(get_agent_graph_html("critique"), unsafe_allow_html=True)
                        final_draft = state.get("draft_answer", "")
                    elif node == "critique":
                        is_hal = state.get('is_hallucination', False)
                        if is_hal:
                            # Handoff back to Synthesize (rewrite)
                            visualizer_container.markdown(get_agent_graph_html("rewrite"), unsafe_allow_html=True)
                            st.toast(f"⚠️ Critic detected hallucination: {state.get('critique', '')}. Rewriting...")
                        else:
                            # Proceeding to End!
                            visualizer_container.markdown(get_agent_graph_html("complete"), unsafe_allow_html=True)
                            st.toast("✅ Critic approved the response!")
            
            st.subheader("Final Synthesized Answer:")
            st.write(final_draft)
            
        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
