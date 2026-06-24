from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from src.agents.state import AgenticRAGState

class SynthesizerAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama(model="llama3")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert engineering assistant. Use the provided retrieved context to answer the user's question. If you don't know the answer or the context doesn't contain it, say so. Do not hallucinate."),
            ("user", "Context: {context}\n\nQuestion: {question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, state: AgenticRAGState) -> dict:
        """
        Synthesizes an answer based solely on the retrieved documents.
        """
        print("--- SYNTHESIZER: Drafting answer ---")
        if not state.get("documents"):
            return {"draft_answer": "I could not find any relevant information in the uploaded documents to answer your question."}
            
        context_str = "\n\n".join([doc.page_content for doc in state["documents"]])
        
        draft = self.chain.invoke({
            "context": context_str,
            "question": state["question"]
        })
        
        return {"draft_answer": draft}
