from typing import List, Dict, Any
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class RAGFusion:
    """Multi-query retrieval and result fusion implementation."""
    
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
    
    def generate_queries(self, question: str, num_queries: int = 3) -> List[str]:
        """Generate multiple query variations from the original question."""
        prompt = PromptTemplate(
            input_variables=["question", "num_queries"],
            template="""
            Generate {num_queries} different versions of the following question. 
            Make each version focus on different aspects or use different wording.
            
            Original question: {question}
            
            Output only the questions, one per line:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(question=question, num_queries=num_queries)
        
        # Parse result into list of queries
        queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
        # Always include the original question
        if question not in queries:
            queries.append(question)
            
        return queries
    
    def reciprocal_rank_fusion(self, document_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Combine multiple document result lists using Reciprocal Rank Fusion.
        
        Args:
            document_lists: List of document result lists to fuse
            k: Constant to prevent division by very small numbers
            
        Returns:
            Fused and reranked list of documents
        """
        # Track document scores in a dictionary
        doc_scores = {}
        
        # Process each result list
        for doc_list in document_lists:
            for rank, doc in enumerate(doc_list):
                doc_id = doc["id"]  # Assuming each doc has a unique ID
                
                # RRF formula: 1 / (rank + k)
                score = 1.0 / (rank + k)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0}
                    
                doc_scores[doc_id]["score"] += score
        
        # Sort documents by score in descending order
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Return just the documents
        return [item["doc"] for item in sorted_docs]
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Perform RAG-Fusion: Generate multiple queries, retrieve, and fuse results."""
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        # Generate multiple query variations
        queries = self.generate_queries(question)
        
        # Retrieve documents for each query
        all_retrieval_results = []
        for query in queries:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            # Convert to consistent format for fusion
            formatted_results = [
                {"id": str(i), "content": doc.page_content, "metadata": doc.metadata, "score": score}
                for i, (doc, score) in enumerate(results)
            ]
            all_retrieval_results.append(formatted_results)
        
        # Fuse results using reciprocal rank fusion
        fused_docs = self.reciprocal_rank_fusion(all_retrieval_results)
        
        # Generate answer based on fused documents
        context = "\n\n".join([doc["content"] for doc in fused_docs[:top_k]])
        
        answer_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Answer the question based on the following context:
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=answer_prompt)
        answer = chain.run(context=context, question=question)
        
        return {
            "answer": answer,
            "source_documents": fused_docs[:top_k]
        }
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        pass

# Example usage
if __name__ == "__main__":
    rag = RAGFusion(documents_path="./data")
    rag.index_documents()
    result = rag.query("What are the applications of transformer models in NLP?")
    print(result["answer"])
