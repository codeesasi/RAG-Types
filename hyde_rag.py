from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

class HyDERAG:
    """
    Hypothetical Document Embeddings (HyDE) RAG implementation.
    Uses hypothetical documents generated by an LLM to improve retrieval for complex queries.
    """
    
    def __init__(self, documents_path: str):
        self.embedding_model = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.2)
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
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would perfectly answer the query.
        
        Args:
            query: The user's query
            
        Returns:
            A hypothetical document text
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Your task is to write a short document that would perfectly answer the following query.
            Imagine you are creating the ideal document that contains exactly the information needed.
            Make it factual, clear, and directly addressing the query.
            Don't include phrases like "according to the document" or references to yourself.
            Just write the document content as if it were an encyclopedia or textbook entry.
            
            Query: {query}
            
            Hypothetical Document:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        hypothetical_doc = chain.run(query=query)
        
        return hypothetical_doc.strip()
    
    def query(self, question: str, top_k: int = 5, use_hyde: bool = True) -> Dict:
        """
        Process a query using HyDE approach.
        
        Args:
            question: The user's query
            top_k: Number of documents to retrieve
            use_hyde: Whether to use HyDE or fall back to direct retrieval
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        if use_hyde:
            # Step 1: Generate a hypothetical document
            hypothetical_doc = self.generate_hypothetical_document(question)
            
            # Step 2: Use the hypothetical document for retrieval instead of the query
            # Convert hypothetical document to a Document object
            hyde_doc = Document(page_content=hypothetical_doc)
            
            # Use the vector store to find documents similar to the hypothetical one
            docs = self.vector_store.similarity_search_by_document(
                document=hyde_doc,
                k=top_k
            )
        else:
            # Fallback to standard retrieval
            docs = self.vector_store.similarity_search(question, k=top_k)
        
        # Step 3: Generate answer using the retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Answer the following question based on the provided context.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=question)
        
        return {
            "answer": answer.strip(),
            "hypothetical_document": hypothetical_doc if use_hyde else None,
            "source_documents": docs
        }
    
    def compare_retrieval_methods(self, question: str, top_k: int = 5) -> Dict:
        """
        Compare HyDE retrieval with direct retrieval.
        
        Args:
            question: The user's query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with results from both methods
        """
        # Get results using HyDE
        hyde_results = self.query(question, top_k=top_k, use_hyde=True)
        
        # Get results using direct retrieval
        direct_results = self.query(question, top_k=top_k, use_hyde=False)
        
        # Analyze overlap between retrieved documents
        hyde_doc_ids = [id(doc) for doc in hyde_results["source_documents"]]
        direct_doc_ids = [id(doc) for doc in direct_results["source_documents"]]
        
        common_docs = len(set(hyde_doc_ids).intersection(set(direct_doc_ids)))
        total_unique_docs = len(set(hyde_doc_ids + direct_doc_ids))
        
        overlap_percentage = (common_docs / total_unique_docs) * 100 if total_unique_docs > 0 else 0
        
        return {
            "hyde_answer": hyde_results["answer"],
            "direct_answer": direct_results["answer"],
            "hypothetical_document": hyde_results["hypothetical_document"],
            "overlap_percentage": overlap_percentage,
            "hyde_documents": hyde_results["source_documents"],
            "direct_documents": direct_results["source_documents"]
        }
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        # This would typically use langchain document loaders
        pass

# Example usage
if __name__ == "__main__":
    rag = HyDERAG(documents_path="./data")
    rag.index_documents()
    
    # Example query that might benefit from HyDE
    query = "What are the economic implications of renewable energy transition for fossil fuel dependent economies?"
    
    # Compare HyDE vs direct retrieval
    comparison = rag.compare_retrieval_methods(query)
    
    print("HYPOTHETICAL DOCUMENT:")
    print(comparison["hypothetical_document"])
    print("\nHyDE ANSWER:")
    print(comparison["hyde_answer"])
    print("\nDIRECT RETRIEVAL ANSWER:")
    print(comparison["direct_answer"])
    print(f"\nDocument overlap: {comparison['overlap_percentage']:.1f}%")
