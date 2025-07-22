from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class ConversationalRAG:
    """Memory-aware multi-turn QA system using RAG."""
    
    def __init__(self, documents_path: str):
        self.embedding_model = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.3)
        self.documents_path = documents_path
        self.vector_store = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.conversation_chain = None
    
    def index_documents(self):
        """Create vector embeddings for documents and store them."""
        documents = self._load_documents(self.documents_path)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        # Initialize the conversation chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory
        )
        
        return len(documents)
    
    def query(self, question: str) -> Dict:
        """Process a user query in the context of the conversation history."""
        if not self.conversation_chain:
            raise ValueError("Documents must be indexed before querying")
        
        response = self.conversation_chain({"question": question})
        return {
            "answer": response["answer"],
            "chat_history": self.get_chat_history()
        }
    
    def get_chat_history(self) -> List[Dict]:
        """Return the current conversation history."""
        return self.memory.chat_memory.messages
    
    def clear_history(self):
        """Reset the conversation history."""
        self.memory.clear()
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        pass

# Example usage
if __name__ == "__main__":
    rag = ConversationalRAG(documents_path="./data")
    rag.index_documents()
    
    # First question
    response1 = rag.query("What is transfer learning in NLP?")
    print("Q1: What is transfer learning in NLP?")
    print(f"A1: {response1['answer']}\n")
    
    # Follow-up question that refers to the previous question
    response2 = rag.query("What are some popular models that use this technique?")
    print("Q2: What are some popular models that use this technique?")
    print(f"A2: {response2['answer']}\n")
    
    # Another follow-up
    response3 = rag.query("Can you explain how one of those models works?")
    print("Q3: Can you explain how one of those models works?")
    print(f"A3: {response3['answer']}")
