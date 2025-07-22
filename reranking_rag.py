from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RerankingRAG:
    """RAG implementation with semantic reranking for enhanced accuracy."""
    
    def __init__(self, documents_path: str, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.embedding_model = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.documents_path = documents_path
        self.vector_store = None
        
        # Load reranker model
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model)
        self.reranker_model.eval()  # Set to evaluation mode
        
    def index_documents(self):
        """Create vector embeddings for documents and store them."""
        documents = self._load_documents(self.documents_path)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        return len(documents)
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 3):
        """Rerank retrieved documents using a cross-encoder reranking model."""
        pairs = [(query, doc["page_content"]) for doc in documents]
        
        # Tokenize query-document pairs
        features = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Get relevance scores
        with torch.no_grad():
            scores = self.reranker_model(**features).logits.flatten()
        
        # Combine documents with scores
        doc_score_pairs = list(zip(documents, scores.tolist()))
        
        # Sort by descending score
        reranked_docs = [doc for doc, score in sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)]
        
        # Return top_k documents
        return reranked_docs[:top_k]
    
    def query(self, question: str, initial_k: int = 10, final_k: int = 3) -> Dict:
        """
        Two-stage retrieval: Initial retrieval followed by reranking.
        
        Args:
            question: User query
            initial_k: Number of documents to retrieve initially
            final_k: Number of documents to keep after reranking
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        # Initial retrieval - get more documents than needed
        initial_docs = self.vector_store.similarity_search(
            question, 
            k=initial_k
        )
        
        # Rerank documents
        reranked_docs = self.rerank(question, initial_docs, top_k=final_k)
        
        # Generate answer using reranked documents
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        
        prompt = f"""
        Answer the question based on the following context:
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        answer = self.llm(prompt)
        
        return {
            "answer": answer,
            "source_documents": reranked_docs
        }
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        pass

# Example usage
if __name__ == "__main__":
    rag = RerankingRAG(documents_path="./data")
    rag.index_documents()
    result = rag.query("What is the role of attention mechanisms in transformers?")
    print(result["answer"])
    print("\nSources:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"{i+1}. {doc.metadata.get('source', 'Unknown')}")
