from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.agents.state import AegisRAGState

class SupervisorOutput(BaseModel):
    needs_rag: bool = Field(description="True if the question requires looking up documents/facts to answer accurately. False if it is a casual greeting or a simple question that can be answered directly.")
    direct_answer: str = Field(description="If needs_rag is False, provide the direct answer here. Otherwise, leave empty.")

class SupervisorAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama(model="llama3", format="json")
        self.parser = JsonOutputParser(pydantic_object=SupervisorOutput)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a routing supervisor. Decide if the user's question requires searching an engineering database (RAG). "
                       "If it's a general question or greeting, set 'needs_rag' to false and provide a 'direct_answer'. "
                       "If it requires technical knowledge or looking up documents, set 'needs_rag' to true.\n{format_instructions}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, state: AegisRAGState) -> dict:
        print(f"--- SUPERVISOR: Analyzing question '{state['question']}' ---")
        try:
            result = self.chain.invoke({
                "chat_history": state.get("chat_history", []),
                "question": state["question"],
                "format_instructions": self.parser.get_format_instructions()
            })
            
            if not result.get("needs_rag", True):
                print("--- SUPERVISOR: No RAG needed. Direct answer provided. ---")
                return {"draft_answer": result.get("direct_answer", ""), "needs_rag": False}
            
            print("--- SUPERVISOR: RAG needed. Routing to researcher. ---")
            return {"needs_rag": True}
        except Exception as e:
            print(f"Supervisor failed: {e}. Defaulting to RAG.")
            return {"needs_rag": True}
