import streamlit as st
import streamlit.components.v1 as components
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
from langchain_core.messages import HumanMessage, AIMessage

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

query = st.chat_input("Enter your engineering query (e.g., structural tolerance limits)")

if query:
    if workflow is None:
        st.error("Please configure your API keys to use the selected model.")
        st.stop()
        
    # Display User Query and append to state
    with st.chat_message("user"):
        st.write(query)
    
    st.session_state.chat_history.append(HumanMessage(content=query))
        
    # Stream execution using the Custom HTML/SVG Visualizer
    with st.chat_message("assistant"):
        visualizer_container = st.empty()
        answer_container = st.empty()
        
        try:
            # Initial render: Researcher active
            with visualizer_container:
                components.html(get_agent_graph_html("research"), height=420)
            
            final_draft = ""
            final_docs = []
            for s in workflow.stream(
                query, 
                chat_history=st.session_state.chat_history[:-1],
                config={"configurable": {"stream_container": answer_container}}
            ):
                # 's' is a dictionary keyed by the node name
                for node, state in s.items():
                    if "documents" in state:
                        final_docs = state["documents"]
                    if node == "supervisor":
                        if not state.get("needs_rag", True):
                            final_draft = state.get("draft_answer", "")
                            with visualizer_container:
                                components.html(get_agent_graph_html("complete"), height=420)
                            st.toast("⚡ Supervisor bypassed RAG and provided a direct answer.")
                        else:
                            with visualizer_container:
                                components.html(get_agent_graph_html("research"), height=420)
                    elif node == "research":
                        # Transition to Synthesizer
                        with visualizer_container:
                            components.html(get_agent_graph_html("synthesize"), height=420)
                    elif node == "synthesize":
                        # Transition to Critic
                        with visualizer_container:
                            components.html(get_agent_graph_html("critique"), height=420)
                        final_draft = state.get("draft_answer", "")
                    elif node == "critique":
                        is_hal = state.get('is_hallucination', False)
                        if is_hal:
                            # Handoff back to Synthesize (rewrite)
                            with visualizer_container:
                                components.html(get_agent_graph_html("rewrite"), height=420)
                            st.toast(f"⚠️ Critic detected hallucination: {state.get('critique', '')}. Rewriting...")
                        else:
                            # Proceeding to End!
                            with visualizer_container:
                                components.html(get_agent_graph_html("complete"), height=420)
                            st.toast("✅ Critic approved the response!")
            
            answer_container.empty()
            display_draft = final_draft
            if final_docs:
                references_md = "\n\n**References & Sources:**\n"
                for i, doc in enumerate(final_docs):
                    source_name = doc.metadata.get('source', 'Unknown Document')
                    page = doc.metadata.get('page', 'N/A')
                    references_md += f"- **[{i+1}]** {source_name} (Page {page})\n"
                
                display_draft += references_md

            st.subheader("Final Synthesized Answer:")
            st.write(final_draft)
            
            if final_docs:
                with st.expander("📚 References & Sources"):
                    for i, doc in enumerate(final_docs):
                        source_name = doc.metadata.get('source', 'Unknown Document')
                        page = doc.metadata.get('page', 'N/A')
                        st.markdown(f"**[{i+1}]** {source_name} (Page {page})")

            st.session_state.chat_history.append(AIMessage(content=display_draft))
            
        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
