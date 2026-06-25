from src.agents.state import AegisRAGState
from src.retrieval.hybrid_search import AdvancedRetriever
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

class ResearcherAgent:
    def __init__(self, llm=None, use_hyde: bool = True, vector_store=None):
        self.retriever = AdvancedRetriever(use_hyde=use_hyde, llm=llm, vector_store=vector_store)
        self.search_tool = DuckDuckGoSearchRun()
        self.llm = llm or ChatOllama(model="llama3")
        
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}")
        ])
        self.contextualize_q_chain = self.contextualize_q_prompt | self.llm | StrOutputParser()

    def invoke(self, state: AegisRAGState) -> dict:
        """
        Takes the current state, retrieves relevant documents using the advanced hybrid search.
        If no documents are found, it falls back to a DuckDuckGo web search.
        """
        q = state["question"]
        chat_history = state.get("chat_history", [])
        
        if chat_history:
            print(f"--- RESEARCHER: Contextualizing query... ---")
            try:
                q = self.contextualize_q_chain.invoke({
                    "chat_history": chat_history,
                    "question": q
                })
                print(f"--- RESEARCHER: Reformulated question to: '{q}' ---")
            except Exception as e:
                print(f"Failed to contextualize query: {e}")
                
        print(f"--- RESEARCHER: Retrieving context for '{q}' ---")
        docs = self.retriever.get_relevant_documents(q)
        
        if not docs:
            print("--- RESEARCHER: No local context found. Falling back to Web Search ---")
            try:
                search_result = self.search_tool.invoke(q)
                docs = [{"page_content": search_result, "metadata": {"source": "DuckDuckGo Web Search"}}]
            except Exception as e:
                print(f"Web search failed: {e}")
                docs = []
        else:
            # Sanitize metadata to ensure msgpack compatibility (e.g. numpy.float32 from Flashrank)
            clean_docs = []
            for d in docs:
                clean_meta = {}
                for k, v in d.metadata.items():
                    if hasattr(v, "item"): # Check if it's a numpy scalar
                        clean_meta[k] = v.item()
                    else:
                        clean_meta[k] = v
                clean_docs.append({"page_content": d.page_content, "metadata": clean_meta})
            docs = clean_docs
        
        return {"documents": docs}
