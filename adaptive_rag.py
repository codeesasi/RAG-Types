from typing import List, Dict, Any, Tuple
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time

class AdaptiveRAG:
    """
    Adaptive RAG implementation that dynamically selects retrieval strategies
    based on query characteristics.
    """
    
    def __init__(self, documents_path: str):
        self.embedding_model = OpenAIEmbeddings()
        # Multiple LLM configurations for different scenarios
        self.fast_llm = OpenAI(temperature=0, max_tokens=256)  # For simple queries
        self.balanced_llm = OpenAI(temperature=0.3, max_tokens=512)  # Default
        self.thorough_llm = OpenAI(temperature=0.7, max_tokens=1024)  # For complex queries
        
        self.documents_path = documents_path
        self.vector_store = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.documents = []
        
        # Performance tracking for adaptive learning
        self.query_history = []
        self.strategy_performance = {
            "vector_only": {"success_rate": 0.5, "avg_latency": 1.0},
            "keyword_only": {"success_rate": 0.5, "avg_latency": 0.8},
            "hybrid": {"success_rate": 0.5, "avg_latency": 1.2},
            "reranked": {"success_rate": 0.6, "avg_latency": 1.5},
        }
        
    def index_documents(self):
        """Create vector embeddings and keyword indices for documents."""
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
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query characteristics to determine the best retrieval strategy.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with query characteristics and recommended strategy
        """
        # Calculate basic query features
        query_length = len(query.split())
        has_specialized_terms = any(len(word) > 10 for word in query.split())
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        is_question = query.strip().endswith('?') or any(query.lower().startswith(w) for w in question_words)
        
        # Calculate query complexity
        complexity_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Rate the complexity of the following query on a scale from 1 to 10, where:
            - 1 means a simple factoid question
            - 5 means a moderately complex question requiring synthesis of multiple facts
            - 10 means a very complex question requiring deep analysis, multiple perspectives, or specialized knowledge
            
            Query: {query}
            
            Provide only a numerical rating from 1-10.
            """
        )
        
        try:
            chain = LLMChain(llm=self.fast_llm, prompt=complexity_prompt)
            complexity_result = chain.run(query=query)
            complexity_score = int(complexity_result.strip())
        except:
            # Default if parsing fails
            complexity_score = 5
        
        # Determine best strategy based on query characteristics
        if complexity_score <= 3:
            # Simple queries: Use fast keyword search for efficiency
            recommended_strategy = "keyword_only"
            recommended_k = 3
            recommended_llm = "fast"
        elif complexity_score <= 6:
            # Moderate queries: Use balanced hybrid approach
            recommended_strategy = "hybrid"
            recommended_k = 5
            recommended_llm = "balanced"
        else:
            # Complex queries: Use thorough reranking
            recommended_strategy = "reranked"
            recommended_k = 8
            recommended_llm = "thorough"
        
        return {
            "query_length": query_length,
            "has_specialized_terms": has_specialized_terms,
            "is_question": is_question,
            "complexity_score": complexity_score,
            "recommended_strategy": recommended_strategy,
            "recommended_k": recommended_k,
            "recommended_llm": recommended_llm
        }
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform keyword-based search using TF-IDF."""
        if self.tfidf_matrix is None:
            raise ValueError("Documents must be indexed before searching")
        
        # Transform query to TF-IDF space
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity between query and documents
        cosine_similarities = np.dot(query_vector, self.tfidf_matrix.T).toarray().flatten()
        
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
        """Perform vector search using embeddings."""
        if not self.vector_store:
            raise ValueError("Documents must be indexed before searching")
        
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        return [{"document": doc, "score": score} for doc, score in results]
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """Combine keyword and vector search results."""
        # Get results from both methods
        keyword_results = self.keyword_search(query, top_k=top_k*2)
        vector_results = self.vector_search(query, top_k=top_k*2)
        
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Process keyword results
        for item in keyword_results:
            doc_id = id(item["document"])
            combined_scores[doc_id] = {
                "document": item["document"],
                "keyword_score": item["score"],
                "vector_score": 0
            }
        
        # Process vector results
        for item in vector_results:
            doc_id = id(item["document"])
            if doc_id in combined_scores:
                combined_scores[doc_id]["vector_score"] = 1.0 - min(item["score"], 1.0)  # Invert if lower is better
            else:
                combined_scores[doc_id] = {
                    "document": item["document"],
                    "keyword_score": 0,
                    "vector_score": 1.0 - min(item["score"], 1.0)  # Invert if lower is better
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
    
    def rerank_results(self, query: str, initial_results: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Rerank results using a cross-encoder approach (simulated with LLM).
        
        Args:
            query: The user's query
            initial_results: Initial retrieved documents
            top_k: Number of documents to keep after reranking
            
        Returns:
            Reranked documents
        """
        if not initial_results:
            return []
        
        reranked_results = []
        
        # For each document, have the LLM evaluate its relevance to the query
        for item in initial_results:
            doc = item["document"]
            
            relevance_prompt = f"""
            On a scale from 0 to 10, how relevant is this document to the query?
            
            Query: {query}
            
            Document: {doc.page_content[:500]}... (truncated)
            
            Rate relevance (0-10):
            """
            
            try:
                relevance_score = float(self.fast_llm(relevance_prompt).strip())
                normalized_score = min(max(relevance_score / 10, 0), 1)
            except:
                normalized_score = 0.5  # Default if parsing fails
            
            reranked_results.append({
                "document": doc,
                "score": normalized_score
            })
        
        # Sort by relevance score
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k results
        return reranked_results[:top_k]
    
    def execute_strategy(self, query: str, strategy: str, k: int) -> Tuple[List[Dict], float]:
        """
        Execute a specific retrieval strategy and measure performance.
        
        Args:
            query: The user's query
            strategy: Retrieval strategy to use
            k: Number of documents to retrieve
            
        Returns:
            Retrieved documents and execution time
        """
        start_time = time.time()
        
        if strategy == "vector_only":
            results = self.vector_search(query, top_k=k)
        elif strategy == "keyword_only":
            results = self.keyword_search(query, top_k=k)
        elif strategy == "hybrid":
            results = self.hybrid_search(query, top_k=k)
        elif strategy == "reranked":
            # First get initial results, then rerank
            initial_results = self.hybrid_search(query, top_k=k*2)
            results = self.rerank_results(query, initial_results, top_k=k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        execution_time = time.time() - start_time
        
        return results, execution_time
    
    def select_llm(self, llm_type: str):
        """Select the appropriate LLM based on query complexity."""
        if llm_type == "fast":
            return self.fast_llm
        elif llm_type == "thorough":
            return self.thorough_llm
        else:  # Default to balanced
            return self.balanced_llm
    
    def query(self, question: str, force_strategy: str = None) -> Dict:
        """
        Process a query using the adaptive RAG approach.
        
        Args:
            question: The user's query
            force_strategy: Optional strategy to force instead of adaptive selection
            
        Returns:
            Dictionary with answer and execution details
        """
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        # Step 1: Analyze query to determine best strategy
        query_analysis = self.analyze_query(question)
        
        # Use forced strategy if provided
        strategy = force_strategy if force_strategy else query_analysis["recommended_strategy"]
        k = query_analysis["recommended_k"]
        llm_type = query_analysis["recommended_llm"]
        
        # Step 2: Execute the selected strategy
        results, execution_time = self.execute_strategy(question, strategy, k)
        
        if not results:
            return {"answer": "No relevant information found.", "execution_time": execution_time}
        
        # Step 3: Generate answer using the appropriate LLM
        llm = self.select_llm(llm_type)
        
        # Prepare context from search results
        context = "\n\n".join([result["document"].page_content for result in results])
        
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
        
        chain = LLMChain(llm=llm, prompt=prompt)
        answer = chain.run(context=context, question=question)
        
        # Record this query in history for future adaptation
        self.query_history.append({
            "query": question,
            "strategy": strategy,
            "execution_time": execution_time,
            "complexity_score": query_analysis["complexity_score"]
        })
        
        return {
            "answer": answer.strip(),
            "strategy_used": strategy,
            "execution_time": execution_time,
            "query_analysis": query_analysis,
            "source_documents": [result["document"] for result in results]
        }
    
    def update_strategy_performance(self, feedback: Dict):
        """
        Update strategy performance metrics based on feedback.
        
        Args:
            feedback: Dictionary with strategy and success flag
        """
        strategy = feedback.get("strategy")
        success = feedback.get("success", True)
        latency = feedback.get("latency")
        
        if strategy and strategy in self.strategy_performance:
            # Update success rate with exponential moving average
            current_rate = self.strategy_performance[strategy]["success_rate"]
            if success:
                new_rate = 0.9 * current_rate + 0.1 * 1.0  # Success
            else:
                new_rate = 0.9 * current_rate + 0.1 * 0.0  # Failure
            
            self.strategy_performance[strategy]["success_rate"] = new_rate
            
            # Update latency if provided
            if latency:
                current_latency = self.strategy_performance[strategy]["avg_latency"]
                new_latency = 0.9 * current_latency + 0.1 * latency
                self.strategy_performance[strategy]["avg_latency"] = new_latency
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        # This would typically use langchain document loaders
        pass

# Example usage
if __name__ == "__main__":
    rag = AdaptiveRAG(documents_path="./data")
    rag.index_documents()
    
    # Simple query
    simple_query = "What is the capital of France?"
    simple_result = rag.query(simple_query)
    print(f"Simple Query Strategy: {simple_result['strategy_used']}")
    print(f"Execution Time: {simple_result['execution_time']:.2f}s")
    print(f"Answer: {simple_result['answer']}")
    
    print("\n" + "-"*50 + "\n")
    
    # Complex query
    complex_query = "What are the potential long-term economic and social implications of widespread automation and artificial intelligence adoption in the manufacturing sector, and how might governments and industries best prepare for these changes?"
    complex_result = rag.query(complex_query)
    print(f"Complex Query Strategy: {complex_result['strategy_used']}")
    print(f"Execution Time: {complex_result['execution_time']:.2f}s")
    print(f"Answer: {complex_result['answer']}")
    
    # Provide feedback to improve adaptive selection
    rag.update_strategy_performance({
        "strategy": complex_result["strategy_used"],
        "success": True,
        "latency": complex_result["execution_time"]
    })
