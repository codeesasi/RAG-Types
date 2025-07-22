from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class SelfRAG:
    """LLM-formulated queries to guide its own retrieval."""
    
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
    
    def generate_retrieval_queries(self, question: str, num_queries: int = 3) -> List[str]:
        """Generate multiple search queries to retrieve relevant information."""
        prompt = PromptTemplate(
            input_variables=["question", "num_queries"],
            template="""
            I need to answer the following question: {question}
            
            To answer this question thoroughly, I need to break it down into {num_queries} 
            specific search queries that will help me retrieve the most relevant information.
            
            Generate {num_queries} different search queries that:
            1. Cover different aspects of the question
            2. Use different phrasings and synonyms
            3. Are specific enough to retrieve relevant information
            4. Together cover all the information needed to answer the original question
            
            Output the queries as a numbered list, one per line:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(question=question, num_queries=num_queries)
        
        # Parse queries
        queries = []
        for line in result.strip().split("\n"):
            # Remove numbering and leading/trailing whitespace
            query = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if query:
                queries.append(query)
        
        # Always include the original question
        if question not in queries:
            queries.append(question)
            
        return queries
    
    def evaluate_relevance(self, query: str, document: str) -> float:
        """Have the LLM evaluate the relevance of a document to the query."""
        prompt = f"""
        On a scale of 0 to 10, rate how relevant the following document is to the query.
        
        Query: {query}
        
        Document: {document}
        
        Provide only a numerical score from 0 to 10, where:
        - 0 means completely irrelevant
        - 10 means perfectly answers the query
        
        Score:
        """
        
        try:
            response = self.llm(prompt)
            score = float(response.strip())
            return min(max(score, 0), 10) / 10  # Normalize to 0-1
        except:
            return 0.5  # Default to middle score if parsing fails
    
    def determine_if_more_info_needed(self, question: str, current_context: str) -> bool:
        """Determine if more information is needed to answer the question."""
        prompt = f"""
        Based on the information I have so far, do I need additional information to provide 
        a complete and accurate answer to the question?
        
        Question: {question}
        
        Information I already have:
        {current_context}
        
        Do I need more information? Answer Yes or No, followed by a brief explanation.
        """
        
        response = self.llm(prompt).strip().lower()
        return "yes" in response.split()[0]
    
    def query(self, question: str, max_iterations: int = 3) -> Dict:
        """
        Perform Self-RAG: generate queries, retrieve, evaluate, and decide if more info is needed.
        
        Args:
            question: The user's question
            max_iterations: Maximum number of retrieval iterations
            
        Returns:
            Dictionary with answer and retrieval process details
        """
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        retrieved_documents = []
        iterations = []
        
        # First iteration
        queries = self.generate_retrieval_queries(question)
        
        for iteration in range(max_iterations):
            iteration_info = {
                "iteration": iteration + 1,
                "queries": queries,
                "documents": []
            }
            
            # Retrieve documents for each query
            newly_retrieved = []
            for query in queries:
                results = self.vector_store.similarity_search(query, k=2)
                
                for doc in results:
                    # Check if document is already retrieved
                    if doc.page_content in [d.page_content for d in retrieved_documents]:
                        continue
                        
                    # Evaluate relevance
                    relevance = self.evaluate_relevance(question, doc.page_content)
                    
                    if relevance > 0.5:  # Only keep if somewhat relevant
                        doc.metadata["relevance"] = relevance
                        doc.metadata["retrieved_by_query"] = query
                        newly_retrieved.append(doc)
                        
                        iteration_info["documents"].append({
                            "content": doc.page_content[:100] + "...",  # Truncated for readability
                            "relevance": relevance,
                            "query": query
                        })
            
            # Add new documents to our collection
            retrieved_documents.extend(newly_retrieved)
            iterations.append(iteration_info)
            
            # Check if we have enough information
            if retrieved_documents:
                current_context = "\n\n".join([doc.page_content for doc in retrieved_documents])
                if not self.determine_if_more_info_needed(question, current_context) or iteration == max_iterations - 1:
                    break
                    
            # Generate new queries based on what we've learned
            if iteration < max_iterations - 1:
                context_so_far = "\n\n".join([doc.page_content for doc in retrieved_documents])
                
                refine_prompt = f"""
                I'm trying to answer the question: {question}
                
                So far, I've gathered the following information:
                {context_so_far}
                
                What additional information do I still need to fully answer the question?
                Generate 2 specific search queries to find this missing information:
                """
                
                refinement = self.llm(refine_prompt)
                new_queries = [q.strip() for q in refinement.split("\n") if q.strip()]
                queries = new_queries[:2]  # Limit to 2 new queries
        
        # Sort documents by relevance
        retrieved_documents.sort(key=lambda x: x.metadata.get("relevance", 0), reverse=True)
        
        # Generate the answer
        if retrieved_documents:
            context = "\n\n".join([doc.page_content for doc in retrieved_documents[:5]])  # Top 5 most relevant
            
            answer_prompt = f"""
            Answer the question based on the following context:
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a comprehensive and accurate answer:
            """
            
            answer = self.llm(answer_prompt)
        else:
            answer = "I couldn't find relevant information to answer this question."
        
        return {
            "answer": answer,
            "source_documents": retrieved_documents,
            "retrieval_process": iterations
        }
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        pass

# Example usage
if __name__ == "__main__":
    import re  # Make sure to import re at the top of your file
    
    rag = SelfRAG(documents_path="./data")
    rag.index_documents()
    
    result = rag.query("What are the economic impacts of climate change on global agriculture?")
    
    print("ANSWER:")
    print(result["answer"])
    
    print("\nRETRIEVAL PROCESS:")
    for iteration in result["retrieval_process"]:
        print(f"\nIteration {iteration['iteration']}:")
        print(f"Queries: {', '.join(iteration['queries'])}")
        print(f"Retrieved {len(iteration['documents'])} documents")
