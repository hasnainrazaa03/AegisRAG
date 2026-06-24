from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from src.agents.state import AegisRAGState

class SynthesizerAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama(model="llama3")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert engineering assistant. Use the provided retrieved context to answer the user's question. When referencing facts, you MUST include inline citations in the format [1], [2], etc., corresponding to the Source numbers provided in the context. If you don't know the answer or the context doesn't contain it, say so. Do not hallucinate."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Context: {context}\n\nQuestion: {question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, state: AegisRAGState, config: RunnableConfig = None) -> dict:
        """
        Synthesizes an answer based solely on the retrieved documents.
        """
        print("--- SYNTHESIZER: Drafting answer ---")
        if not state.get("documents"):
            return {"draft_answer": "I could not find any relevant information in the uploaded documents to answer your question."}
            
        context_str = "\n\n".join([f"[Source {i+1}]\n{doc['page_content']}" for i, doc in enumerate(state["documents"])])
        
        container = config.get("configurable", {}).get("stream_container") if config else None
        
        draft = ""
        if container:
            for chunk in self.chain.stream({
                "chat_history": state.get("chat_history", []),
                "context": context_str,
                "question": state["question"]
            }):
                draft += chunk
                container.markdown(draft)
        else:
            draft = self.chain.invoke({
                "chat_history": state.get("chat_history", []),
                "context": context_str,
                "question": state["question"]
            })
        
        return {"draft_answer": draft}
