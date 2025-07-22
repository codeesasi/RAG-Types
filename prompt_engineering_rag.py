from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class PromptEngineeringRAG:
    """RAG with customized prompt templates and context injection."""
    
    def __init__(self, documents_path: str):
        self.embedding_model = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.documents_path = documents_path
        self.vector_store = None
        
    def index_documents(self):
        """Create vector embeddings for documents and store them."""
        documents = self._load_documents(self.documents_path)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        return len(documents)
    
    def query(self, question: str, context: Dict[str, Any] = None, top_k: int = 3) -> str:
        """Retrieve relevant documents and generate an answer with custom prompt."""
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        # Define custom prompt template with context injection
        prompt_template = """
        You are an AI assistant providing accurate information based on the given context.
        
        Context: {context}
        
        {custom_instructions}
        
        Question: {question}
        
        Answer:
        """
        
        # Default custom instructions if none provided
        if not context or "custom_instructions" not in context:
            context = context or {}
            context["custom_instructions"] = "Provide a concise and factual answer."
        
        # Create prompt
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question", "custom_instructions"]
        )
        
        # Set up retriever and chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Run the chain
        response = qa_chain({"question": question, **context})
        return response["result"]
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        pass

# Example usage
if __name__ == "__main__":
    rag = PromptEngineeringRAG(documents_path="./data")
    rag.index_documents()
    
    # Example with custom context
    answer = rag.query(
        "What are the benefits of RAG?",
        context={
            "custom_instructions": "Format your answer as a numbered list and use examples."
        }
    )
    print(answer)
