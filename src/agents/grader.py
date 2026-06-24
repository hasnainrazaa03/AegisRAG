from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from src.agents.state import AegisRAGState

class GraderAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOllama(model="llama3", format="json")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing the relevance of a retrieved document to a user question.\n"
                       "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n"
                       "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n"
                       "Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
        ])
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def invoke(self, state: AegisRAGState) -> dict:
        print("--- GRADER: Grading documents ---")
        question = state["question"]
        documents = state.get("documents", [])
        
        filtered_docs = []
        for doc in documents:
            try:
                score = self.chain.invoke({
                    "question": question,
                    "document": doc.page_content
                })
                if score.get("score", "").lower() == "yes":
                    print("--- GRADER: Document relevant ---")
                    filtered_docs.append(doc)
                else:
                    print("--- GRADER: Document irrelevant ---")
            except Exception as e:
                # If parsing fails, default to keeping the document to be safe
                print(f"--- GRADER: Failed to grade document: {e} ---")
                filtered_docs.append(doc)
                
        return {"documents": filtered_docs}
