from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HybridRAG:
    """RAG system combining keyword (TF-IDF) and vector search."""
    
    def __init__(self, documents_path: str):
        self.embedding_model = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.documents_path = documents_path
        self.vector_store = None
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.documents = []
        
    def index_documents(self):
        """Create both vector embeddings and TF-IDF representations for documents."""
        documents = self._load_documents(self.documents_path)
        self.documents = documents
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        # Create TF-IDF index
        document_texts = [doc.page_content for doc in documents]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(document_texts)
        
        return len(documents)
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform keyword-based search using TF-IDF."""
        if self.tfidf_matrix is None:
            raise ValueError("Documents must be indexed before searching")
        
        # Transform query to TF-IDF space
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity between query and documents
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k document indices
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        
        # Create result list
        results = []
        for idx in top_indices:
            if cosine_similarities[idx] > 0:  # Only include if there's some similarity
                results.append({
                    "document": self.documents[idx],
                    "score": float(cosine_similarities[idx])
                })
        
        return results
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search using vector embeddings."""
        if not self.vector_store:
            raise ValueError("Documents must be indexed before searching")
        
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        return [{"document": doc, "score": score} for doc, score in results]
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """
        Combine keyword and vector search results.
        
        Args:
            query: The search query
            top_k: Number of results to return
            alpha: Weight for vector search (0-1). Keyword weight = 1-alpha
            
        Returns:
            Combined and reranked list of documents
        """
        # Get results from both methods
        keyword_results = self.keyword_search(query, top_k=top_k*2)
        vector_results = self.vector_search(query, top_k=top_k*2)
        
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Normalize keyword scores (higher is better)
        if keyword_results:
            max_keyword_score = max(item["score"] for item in keyword_results)
            min_keyword_score = min(item["score"] for item in keyword_results)
            range_keyword = max_keyword_score - min_keyword_score
            
            for item in keyword_results:
                doc_id = id(item["document"])
                if range_keyword > 0:
                    normalized_score = (item["score"] - min_keyword_score) / range_keyword
                else:
                    normalized_score = 1.0 if item["score"] > 0 else 0.0
                    
                combined_scores[doc_id] = {
                    "document": item["document"],
                    "keyword_score": normalized_score,
                    "vector_score": 0
                }
        
        # Normalize vector scores (lower is better in some implementations, so we invert)
        if vector_results:
            max_vector_score = max(item["score"] for item in vector_results)
            min_vector_score = min(item["score"] for item in vector_results)
            range_vector = max_vector_score - min_vector_score
            
            for item in vector_results:
                doc_id = id(item["document"])
                if range_vector > 0:
                    # Assuming lower scores are better for vector search
                    normalized_score = 1 - ((item["score"] - min_vector_score) / range_vector)
                else:
                    normalized_score = 1.0
                
                if doc_id in combined_scores:
                    combined_scores[doc_id]["vector_score"] = normalized_score
                else:
                    combined_scores[doc_id] = {
                        "document": item["document"],
                        "keyword_score": 0,
                        "vector_score": normalized_score
                    }
        
        # Calculate weighted scores
        for doc_id in combined_scores:
            item = combined_scores[doc_id]
            item["combined_score"] = (alpha * item["vector_score"] + 
                                     (1 - alpha) * item["keyword_score"])
        
        # Sort by combined score and get top-k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]
        
        return [{"document": item["document"], "score": item["combined_score"]} 
                for item in sorted_results]
    
    def query(self, question: str, top_k: int = 3, alpha: float = 0.5) -> Dict:
        """Process a user query using hybrid search."""
        search_results = self.hybrid_search(question, top_k=top_k, alpha=alpha)
        
        if not search_results:
            return {"answer": "No relevant information found."}
        
        # Prepare context from search results
        context = "\n\n".join([result["document"].page_content for result in search_results])
        
        # Generate answer
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
            "source_documents": [result["document"] for result in search_results]
        }
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        pass

# Example usage
if __name__ == "__main__":
    rag = HybridRAG(documents_path="./data")
    rag.index_documents()
    
    # Try with different alpha values to see the effect
    result1 = rag.query("What are the key components of a transformer model?", alpha=0.7)
    print("Vector-leaning (alpha=0.7):")
    print(result1["answer"])
    
    result2 = rag.query("What are the key components of a transformer model?", alpha=0.3)
    print("\nKeyword-leaning (alpha=0.3):")
    print(result2["answer"])
