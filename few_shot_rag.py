from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class FewShotRAG:
    """
    Few-Shot RAG implementation that incorporates examples to guide the LLM
    on how to use retrieved information effectively.
    """
    
    def __init__(self, documents_path: str, examples_path: str = None):
        self.embedding_model = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.3)
        self.documents_path = documents_path
        self.examples_path = examples_path
        self.vector_store = None
        self.examples = []
        
    def index_documents(self):
        """Create vector embeddings for documents and store them."""
        documents = self._load_documents(self.documents_path)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        # Load examples if available
        if self.examples_path:
            self.examples = self._load_examples(self.examples_path)
            
        return len(documents)
    
    def _load_examples(self, examples_path: str) -> List[Dict]:
        """
        Load example query-document-answer triplets.
        
        In a real implementation, this would load from files or a database.
        Here, we'll use hardcoded examples for demonstration.
        """
        # Placeholder for demonstration - in a real system, load from files
        examples = [
            {
                "query": "What are the main components of a transformer model?",
                "documents": [
                    "Transformer models consist of several key components: (1) Input embeddings that convert tokens to vectors, (2) Positional encodings that add information about token position, (3) Multi-head attention mechanisms that compute relationships between tokens, (4) Feed-forward neural networks that process each position, (5) Layer normalization for training stability, and (6) Residual connections to prevent gradient vanishing."
                ],
                "answer": "The main components of a transformer model include:\n1. Input embeddings - convert tokens to vector representations\n2. Positional encodings - add information about token positions\n3. Multi-head attention mechanisms - compute relationships between tokens\n4. Feed-forward neural networks - process each position independently\n5. Layer normalization - ensures training stability\n6. Residual connections - helps prevent vanishing gradients\n\nThese components work together to process sequential data in parallel, allowing transformers to capture long-range dependencies efficiently."
            },
            {
                "query": "How does transfer learning work in NLP?",
                "documents": [
                    "Transfer learning in NLP involves pre-training a language model on a large corpus of text using self-supervised objectives like masked language modeling or next token prediction. This allows the model to learn general language patterns, grammar, and some world knowledge. The pre-trained model is then fine-tuned on a smaller, task-specific dataset with labeled examples. During fine-tuning, the model parameters are updated to optimize performance on the specific task while retaining the general language understanding from pre-training."
                ],
                "answer": "Transfer learning in NLP works through a two-stage process:\n\nFirst, in the pre-training stage, a language model is trained on vast amounts of text using self-supervised objectives (like predicting masked words or the next token). This teaches the model general language patterns and knowledge without requiring labeled data.\n\nSecond, in the fine-tuning stage, this pre-trained model is adapted to a specific task using a smaller labeled dataset. This approach is powerful because:\n\n- It leverages knowledge from massive text corpora\n- Reduces the need for large task-specific datasets\n- Significantly improves performance on downstream tasks\n- Decreases training time and computational resources\n\nThis paradigm has revolutionized NLP by enabling state-of-the-art performance across many applications."
            }
        ]
        return examples
    
    def retrieve_similar_examples(self, query: str, num_examples: int = 2) -> List[Dict]:
        """
        Retrieve examples that are similar to the current query.
        
        Args:
            query: The user's query
            num_examples: Number of examples to retrieve
            
        Returns:
            List of relevant examples
        """
        if not self.examples:
            return []
            
        # In a real implementation, you would use vector similarity to find examples
        # For simplicity, we'll just return the first n examples
        return self.examples[:min(num_examples, len(self.examples))]
    
    def format_examples(self, examples: List[Dict]) -> str:
        """Format examples for inclusion in the prompt."""
        if not examples:
            return ""
            
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            formatted_example = f"""
            EXAMPLE {i}:
            
            Query: {example['query']}
            
            Retrieved Information:
            {example['documents'][0]}
            
            Answer:
            {example['answer']}
            """
            formatted_examples.append(formatted_example)
            
        return "\n\n".join(formatted_examples)
    
    def query(self, question: str, top_k: int = 5, num_examples: int = 2) -> Dict:
        """
        Process a query using few-shot learning approach.
        
        Args:
            question: The user's query
            top_k: Number of documents to retrieve
            num_examples: Number of examples to include
            
        Returns:
            Dictionary with answer and source information
        """
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        # Step 1: Retrieve similar documents
        docs = self.vector_store.similarity_search(question, k=top_k)
        
        # Step 2: Retrieve similar examples
        examples = self.retrieve_similar_examples(question, num_examples)
        
        # Step 3: Format examples and retrieved documents
        formatted_examples = self.format_examples(examples)
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Step 4: Generate answer with few-shot prompting
        prompt = PromptTemplate(
            input_variables=["examples", "retrieved_context", "question"],
            template="""
            You are a knowledgeable assistant that provides detailed, accurate answers.
            Below are some examples of how to effectively use retrieved information to answer questions.
            Follow these examples to craft your own answer to the new query.
            
            {examples}
            
            Now, answer the following query using the retrieved information:
            
            Query: {question}
            
            Retrieved Information:
            {retrieved_context}
            
            Answer:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(
            examples=formatted_examples,
            retrieved_context=retrieved_context,
            question=question
        )
        
        return {
            "answer": answer.strip(),
            "source_documents": docs,
            "examples_used": len(examples)
        }
    
    def add_example(self, query: str, documents: List[str], answer: str):
        """
        Add a new example to the examples collection.
        
        Args:
            query: The example query
            documents: The retrieved documents for this query
            answer: The high-quality answer
        """
        new_example = {
            "query": query,
            "documents": documents,
            "answer": answer
        }
        
        self.examples.append(new_example)
        print(f"Added new example: '{query}'")
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        # This would typically use langchain document loaders
        pass

# Example usage
if __name__ == "__main__":
    rag = FewShotRAG(documents_path="./data", examples_path="./examples")
    rag.index_documents()
    
    # Query with few-shot learning
    query = "Explain how attention mechanisms work in deep learning"
    result = rag.query(query)
    
    print("QUERY:")
    print(query)
    print("\nANSWER:")
    print(result["answer"])
    print(f"\nExamples used: {result['examples_used']}")
    
    # Add this as a new example (in a real system, you'd do this for high-quality responses)
    if result["examples_used"] > 0:  # Only if we had examples to begin with
        rag.add_example(
            query=query,
            documents=[doc.page_content for doc in result["source_documents"]],
            answer=result["answer"]
        )
