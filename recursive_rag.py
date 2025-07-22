from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class RecursiveRAG:
    """
    Recursive RAG implementation that chains multiple RAG processes together
    for complex multi-hop reasoning.
    """
    
    def __init__(self, documents_path: str):
        self.embedding_model = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.3)
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
    
    def decompose_question(self, question: str) -> List[Dict]:
        """
        Break down a complex question into a sequence of simpler sub-questions.
        
        Args:
            question: The complex user query
            
        Returns:
            List of sub-questions with dependencies
        """
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an expert at breaking down complex questions into simpler, sequential sub-questions.
            
            For the following complex question, identify 3-5 sub-questions that would need to be answered in sequence
            to fully address the original question. Each sub-question should:
            
            1. Be simpler than the original question
            2. Build logically towards answering the original question
            3. Potentially depend on answers to previous sub-questions
            
            Complex Question: {question}
            
            For each sub-question, provide:
            1. The sub-question text
            2. A list of previous sub-question numbers it depends on (if any)
            
            Format your response as a JSON-like structure:
            [
                {{"id": 1, "question": "sub-question 1", "depends_on": []}},
                {{"id": 2, "question": "sub-question 2", "depends_on": [1]}},
                {{"id": 3, "question": "sub-question 3", "depends_on": [1, 2]}},
                ...
            ]
            
            Only output the JSON-like structure without any additional text.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(question=question)
        
        # Parse the result into a list of dictionaries
        # In a real implementation, you would use json.loads() with proper error handling
        # This is a simplified parsing approach
        import json
        try:
            sub_questions = json.loads(result)
            return sub_questions
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            # In a real implementation, you would add more robust parsing logic
            return [{"id": 1, "question": question, "depends_on": []}]
    
    def answer_sub_question(self, sub_question: str, context: str = "", top_k: int = 3) -> str:
        """
        Answer a single sub-question using RAG.
        
        Args:
            sub_question: The sub-question to answer
            context: Additional context from previous answers
            top_k: Number of documents to retrieve
            
        Returns:
            Answer to the sub-question
        """
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(sub_question, k=top_k)
        doc_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Combine retrieved documents with previous context
        full_context = f"{context}\n\n{doc_context}" if context else doc_context
        
        # Generate answer using the context
        prompt = PromptTemplate(
            input_variables=["context", "sub_question"],
            template="""
            Answer the following question based on the provided context.
            If the context doesn't contain the information needed, say "I cannot determine this from the available information."
            
            Context:
            {context}
            
            Question: {sub_question}
            
            Answer:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=full_context, sub_question=sub_question)
        
        return answer.strip()
    
    def synthesize_final_answer(self, original_question: str, sub_questions: List[Dict], sub_answers: Dict[int, str]) -> str:
        """
        Synthesize a final answer from all the sub-question answers.
        
        Args:
            original_question: The original complex question
            sub_questions: List of sub-questions with their dependencies
            sub_answers: Dictionary mapping sub-question IDs to their answers
            
        Returns:
            Synthesized final answer
        """
        # Prepare the context with all sub-questions and answers
        context_parts = []
        for sq in sub_questions:
            q_id = sq["id"]
            q_text = sq["question"]
            q_answer = sub_answers.get(q_id, "No answer available")
            context_parts.append(f"Sub-question {q_id}: {q_text}\nAnswer: {q_answer}")
        
        context = "\n\n".join(context_parts)
        
        # Generate synthesized answer
        prompt = PromptTemplate(
            input_variables=["original_question", "context"],
            template="""
            You are tasked with synthesizing a comprehensive answer to a complex question.
            You have been provided with answers to several sub-questions that break down the original complex question.
            
            Original Complex Question: {original_question}
            
            Information from sub-questions:
            {context}
            
            Please synthesize a complete, coherent answer to the original question.
            Your answer should:
            1. Directly address the original question
            2. Integrate information from all relevant sub-questions
            3. Present a logical flow of reasoning
            4. Be comprehensive yet concise
            
            Synthesized Answer:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        final_answer = chain.run(original_question=original_question, context=context)
        
        return final_answer.strip()
    
    def query(self, question: str) -> Dict:
        """
        Process a complex query through recursive RAG.
        
        Args:
            question: The complex user query
            
        Returns:
            Dictionary with final answer and reasoning process
        """
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        # Step 1: Decompose the question into sub-questions
        sub_questions = self.decompose_question(question)
        
        # Step 2: Answer each sub-question in order, respecting dependencies
        sub_answers = {}
        intermediate_context = ""
        
        for sq in sub_questions:
            q_id = sq["id"]
            q_text = sq["question"]
            dependencies = sq["depends_on"]
            
            # Build context from dependencies
            dep_context = ""
            for dep_id in dependencies:
                if dep_id in sub_answers:
                    dep_sq = next((s for s in sub_questions if s["id"] == dep_id), None)
                    if dep_sq:
                        dep_context += f"Question: {dep_sq['question']}\nAnswer: {sub_answers[dep_id]}\n\n"
            
            # Answer this sub-question with context from dependencies
            answer = self.answer_sub_question(q_text, context=dep_context)
            sub_answers[q_id] = answer
            
            # Add to intermediate context for logging/debugging
            intermediate_context += f"Sub-question {q_id}: {q_text}\nAnswer: {answer}\n\n"
        
        # Step 3: Synthesize final answer
        final_answer = self.synthesize_final_answer(question, sub_questions, sub_answers)
        
        return {
            "answer": final_answer,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
            "reasoning_process": intermediate_context
        }
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        # This would typically use langchain document loaders
        pass

# Example usage
if __name__ == "__main__":
    rag = RecursiveRAG(documents_path="./data")
    rag.index_documents()
    
    complex_question = "How did the development of the printing press impact literacy rates in Europe, and how did this subsequently affect political movements in the 18th century?"
    
    result = rag.query(complex_question)
    
    print("FINAL ANSWER:")
    print(result["answer"])
    print("\nREASONING PROCESS:")
    print(result["reasoning_process"])
