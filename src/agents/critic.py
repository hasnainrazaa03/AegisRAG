from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.agents.state import AgenticRAGState

class CritiqueOutput(BaseModel):
    is_hallucination: bool = Field(description="True if the answer contains information NOT present in the context.")
    critique: str = Field(description="Explanation of why it is or isn't a hallucination.")

class CriticAgent:
    def __init__(self, llm=None):
        # If no LLM is provided, use a local model that is good at following JSON schemas (like llama3)
        if llm is None:
            self.llm = ChatOllama(model="llama3", format="json")
        else:
            self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=CritiqueOutput)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict grading assistant. Evaluate if the Draft Answer is fully supported by the Context. If the draft contains ANY information not in the Context, mark it as a hallucination. Respond ONLY with valid JSON matching the schema.\n{format_instructions}"),
            ("user", "Context: {context}\n\nDraft Answer: {draft_answer}")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, state: AgenticRAGState) -> dict:
        """
        Critiques the draft answer against the retrieved documents.
        """
        print("--- CRITIC: Evaluating draft for hallucinations ---")
        context_str = "\n\n".join([doc.page_content for doc in state["documents"]])
        
        try:
            result = self.chain.invoke({
                "context": context_str,
                "draft_answer": state["draft_answer"],
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return {
                "critique": result.get("critique", ""),
                "is_hallucination": result.get("is_hallucination", False),
                "iterations": state.get("iterations", 0) + 1
            }
        except Exception as e:
            print(f"Critic parsing failed: {e}")
            # Safe fallback: Treat as hallucination to force a rewrite or end if max iterations
            return {"is_hallucination": True, "critique": "Critic failed to parse output, assuming hallucination."}
