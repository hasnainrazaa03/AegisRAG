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
from langchain_google_genai import ChatGoogleGenerativeAI
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
    ("Local (Ollama Llama 3)", "Cloud (Claude 3.5 Sonnet)", "Cloud (Gemini 2.5 Flash)"),
    index=2
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
    elif model_choice == "Cloud (Gemini 2.5 Flash)":
        if config.GOOGLE_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_ollama import ChatOllama
            # Fallback gracefully to local Ollama if Gemini gets rate-limited (429)
            primary = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config.GOOGLE_API_KEY)
            fallback = ChatOllama(model="llama3", temperature=0)
            return primary.with_fallbacks([fallback])
        else:
            st.sidebar.error("⚠️ GOOGLE_API_KEY not found in .env")
            return None
    else:
        return ChatOllama(model="llama3")

llm = get_llm()

from src.retrieval.vector_store import VectorStore

# Initialize VectorStore globally so it doesn't try to open Qdrant multiple times
@st.cache_resource
def get_vector_store():
    return VectorStore()

vector_store = get_vector_store()

@st.cache_resource
def init_folder_watcher(_vector_store):
    from src.ingestion.watcher import start_watcher
    return start_watcher(_vector_store)

watcher = init_folder_watcher(vector_store)

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
                
                # Replace the temporary path with the actual filename for display
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                
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

st.sidebar.divider()

@st.dialog("Knowledge Graph Visualization", width="large")
def render_knowledge_graph():
    with st.spinner("Fetching vectors and computing 3D projection (PCA)..."):
        try:
            points, payloads = workflow.researcher.retriever.vector_store.get_all_vectors()
            if not points:
                st.warning("No documents ingested yet. Please upload a document first.")
                return
                
            st.write(f"Visualizing **{len(points)}** knowledge chunks.")
            
            from sklearn.decomposition import PCA
            import plotly.express as px
            import pandas as pd
            
            pca = PCA(n_components=3)
            components = pca.fit_transform(points)
            
            df = pd.DataFrame(components, columns=['x', 'y', 'z'])
            df['source'] = [p.get('metadata', {}).get('source', 'Unknown') for p in payloads]
            df['content'] = [p.get('page_content', '')[:150] + '...' for p in payloads]
            
            fig = px.scatter_3d(
                df, x='x', y='y', z='z',
                color='source',
                hover_data=['content'],
                title="3D PCA of Knowledge Graph Vectors"
            )
            fig.update_traces(marker=dict(size=5, opacity=0.8))
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render graph: {e}")

if st.sidebar.button("🌌 Visualize Knowledge Graph", use_container_width=True):
    render_knowledge_graph()


def render_live_connections_graph(query_text, retrieved_docs, container=st):
    with st.spinner("Computing 2D AI Inference Connections..."):
        try:
            import numpy as np
            points, payloads = workflow.researcher.retriever.vector_store.get_all_vectors()
            
            # Enforce strict homogeneity to prevent numpy inhomogeneous shape errors
            valid_points = []
            valid_payloads = []
            for p, pay in zip(points, payloads):
                if hasattr(p, '__len__') and len(p) == 384:
                    # Explicitly convert to 1D float array to guarantee homogeneity
                    arr = np.array(p, dtype=float).flatten()
                    if arr.shape == (384,):
                        valid_points.append(arr)
                        valid_payloads.append(pay)
            
            points = valid_points
            payloads = valid_payloads
            
            if not points:
                return
            
            if query_text:
                query_vector = workflow.researcher.retriever.vector_store.embeddings.embed_query(query_text)
                query_arr = np.array(query_vector, dtype=float).flatten()
                if query_arr.shape == (384,):
                    all_vectors = points + [query_arr]
                else:
                    query_text = None # fallback if embed_query shape is weird
                    all_vectors = points
            else:
                query_vector = None
                all_vectors = points
            
            from sklearn.decomposition import PCA
            import plotly.graph_objects as go
            import pandas as pd
            
            # Final safety check before PCA
            all_vectors = np.vstack(all_vectors)
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(all_vectors)
            
            if query_vector:
                bg_components = components[:-1]
                query_comp = components[-1]
            else:
                bg_components = components
                query_comp = None
            
            fig = go.Figure()
            
            # Background points
            fig.add_trace(go.Scatter(
                x=bg_components[:, 0].tolist(),
                y=bg_components[:, 1].tolist(),
                mode='markers',
                marker=dict(size=6, color='lightgray', opacity=0.5),
                name='Knowledge Base'
            ))
            
            retrieved_contents = [d["page_content"] for d in retrieved_docs]
            retrieved_indices = []
            for i, p in enumerate(payloads):
                if p.get('page_content') in retrieved_contents:
                    retrieved_indices.append(i)
                    
            # Draw retrieved points and lines
            if retrieved_indices:
                ret_x = [float(bg_components[i, 0]) for i in retrieved_indices]
                ret_y = [float(bg_components[i, 1]) for i in retrieved_indices]
                
                fig.add_trace(go.Scatter(
                    x=ret_x,
                    y=ret_y,
                    mode='markers',
                    marker=dict(size=10, color='blue', symbol='circle'),
                    name='Retrieved Chunks'
                ))
                
                for rx, ry in zip(ret_x, ret_y):
                    fig.add_trace(go.Scatter(
                        x=[float(query_comp[0]), rx],
                        y=[float(query_comp[1]), ry],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        showlegend=False
                    ))
            
            # Query point
            if query_comp is not None:
                fig.add_trace(go.Scatter(
                    x=[float(query_comp[0])],
                    y=[float(query_comp[1])],
                    mode='markers',
                    marker=dict(size=16, color='red', symbol='star'),
                    name='Your Query'
                ))
            
            fig.update_layout(title="🧠 Live AI Inference Connections (2D PCA)", margin=dict(l=0, r=0, t=30, b=0))
            
            # Generate a unique key using the number of retrieved docs so Streamlit knows it's a new frame in the animation
            unique_key = f"live_graph_{len(retrieved_docs) if retrieved_docs else 0}"
            container.plotly_chart(fig, use_container_width=True, key=unique_key)
            
        except Exception as e:
            st.error(f"Failed to generate connection graph: {e}")

# ---- MAIN INTERFACE ----

from langchain_core.messages import HumanMessage, AIMessage

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False
if "resume_graph" not in st.session_state:
    st.session_state.resume_graph = False
if "pending_docs" not in st.session_state:
    st.session_state.pending_docs = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

disable_chat = st.session_state.awaiting_approval or st.session_state.resume_graph
query = st.chat_input("Enter your engineering query (e.g., structural tolerance limits)", disabled=disable_chat)

if query or st.session_state.resume_graph:
    if workflow is None:
        st.error("Please configure your API keys to use the selected model.")
        st.stop()
        
    if query and not st.session_state.resume_graph:
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.current_query = query
        
    query_to_run = st.session_state.get("current_query", "")
    
    # Display User Query
    with st.chat_message("user"):
        st.write(query_to_run)
        
    # Stream execution using the Custom HTML/SVG Visualizer
    with st.chat_message("assistant"):
        visualizer_container = st.empty()
        answer_container = st.empty()
        
        config = {"configurable": {"thread_id": st.session_state.thread_id, "stream_container": answer_container}}
        
        try:
            if st.session_state.resume_graph:
                with visualizer_container:
                    components.html(get_agent_graph_html("synthesize"), height=420)
                state_info = workflow.app.get_state(config)
                final_docs = state_info.values.get("documents", [])
                stream_input = None
            else:
                with visualizer_container:
                    components.html(get_agent_graph_html("research"), height=420)
                stream_input = query_to_run
                final_docs = []
            
            final_draft = ""
            
            with st.expander("🧠 Live AI Inference Connections", expanded=True):
                graph_container = st.empty()
                if not st.session_state.resume_graph:
                    render_live_connections_graph(None, [], container=graph_container)
                else:
                    state_info = workflow.app.get_state(config)
                    render_live_connections_graph(st.session_state.chat_history[-1].content, state_info.values.get("documents", []), container=graph_container)
            for s in workflow.stream(
                stream_input, 
                chat_history=st.session_state.chat_history[:-1],
                config=config
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
                        with visualizer_container:
                            components.html(get_agent_graph_html("grade"), height=420)
                            
                        # Animate the connections being found one-by-one!
                        import time
                        docs = state.get("documents", [])
                        for i in range(1, len(docs) + 1):
                            render_live_connections_graph(query_to_run, docs[:i], container=graph_container)
                            time.sleep(0.6) # Short pause to see the connection drawn
                    elif node == "grade":
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
            
            # Check if interrupted
            state_info = workflow.app.get_state(config)
            if state_info.next and "synthesize" in state_info.next:
                st.session_state.awaiting_approval = True
                st.session_state.pending_docs = final_docs
                st.rerun()
            else:
                st.session_state.awaiting_approval = False
                st.session_state.resume_graph = False
                
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
                    with st.expander("View References & Sources"):
                        for i, doc in enumerate(final_docs):
                            st.markdown(f"**[{i+1}] {doc.get('metadata', {}).get('source', 'Unknown')} (Page {doc.get('metadata', {}).get('page', 'N/A')})**")
                            st.text(doc.get("page_content", "")[:300] + "...")
                            
                # Create and display download button for the report
                report_content = f"# AegisRAG Research Report\n\n"
                report_content += f"**Question:** {query_to_run}\n\n"
                report_content += f"**Answer:**\n{final_draft}\n\n"
                if final_docs:
                    report_content += references_md
                    
                st.download_button(
                    label="📥 Download Report (.md)",
                    data=report_content,
                    file_name="aegisrag_report.md",
                    mime="text/markdown"
                )
                


                st.session_state.chat_history.append(AIMessage(content=display_draft))
            
        except Exception as e:
            st.error(f"An error occurred during execution: {e}")


if st.session_state.awaiting_approval:
    with st.chat_message("assistant"):
        st.info("⏸️ Execution paused by Human-in-the-Loop.")
        with st.expander("Review Retrieved Context before Synthesis", expanded=True):
            if st.session_state.pending_docs:
                for i, doc in enumerate(st.session_state.pending_docs):
                    st.markdown(f"**[{i+1}] {doc.get('metadata', {}).get('source', 'Unknown')} (Page {doc.get('metadata', {}).get('page', 'N/A')})**")
                    st.write(doc.get("page_content", ""))
            else:
                st.write("No relevant documents found. The Synthesizer will likely fail to answer.")
        
        col1, col2 = st.columns(2)
        if col1.button("✅ Approve & Continue"):
            st.session_state.awaiting_approval = False
            st.session_state.resume_graph = True
            st.rerun()
        if col2.button("❌ Reject & Cancel"):
            st.session_state.awaiting_approval = False
            st.session_state.resume_graph = False
            st.session_state.chat_history.append(AIMessage(content="Search cancelled by user during HIL review."))
            st.rerun()
