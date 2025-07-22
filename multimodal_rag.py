from typing import List, Dict, Any, Union
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import os
import base64
from PIL import Image
import io
import fitz  # PyMuPDF for PDF processing
import pytesseract
from transformers import CLIPProcessor, CLIPModel

class MultimodalRAG:
    """RAG system that handles retrieval from multiple data formats (text, images, PDFs)."""
    
    def __init__(self, documents_path: str, image_path: str = None, pdf_path: str = None):
        self.embedding_model = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.documents_path = documents_path
        self.image_path = image_path
        self.pdf_path = pdf_path
        self.vector_store = None
        
        # For image embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def index_documents(self):
        """Create vector embeddings for all content types."""
        # Process text documents
        text_documents = self._load_documents(self.documents_path)
        
        # Process images if path provided
        image_documents = []
        if self.image_path:
            image_documents = self._process_images(self.image_path)
            
        # Process PDFs if path provided
        pdf_documents = []
        if self.pdf_path:
            pdf_documents = self._process_pdfs(self.pdf_path)
            
        # Combine all documents
        all_documents = text_documents + image_documents + pdf_documents
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=all_documents,
            embedding=self.embedding_model
        )
        
        return len(all_documents)
    
    def _process_images(self, image_path: str) -> List:
        """Process images and convert to document format with extracted text and metadata."""
        documents = []
        
        for filename in os.listdir(image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(image_path, filename)
                
                try:
                    # Open image
                    image = Image.open(file_path)
                    
                    # Extract text using OCR
                    extracted_text = pytesseract.image_to_string(image)
                    
                    # Get CLIP embeddings for the image
                    clip_inputs = self.clip_processor(
                        text=None,
                        images=image,
                        return_tensors="pt"
                    )
                    image_features = self.clip_model.get_image_features(**clip_inputs)
                    
                    # Convert image to base64 for storage
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Create document
                    document = {
                        "page_content": extracted_text if extracted_text.strip() else f"[Image: {filename}]",
                        "metadata": {
                            "source": file_path,
                            "type": "image",
                            "image_data": img_str,
                            "filename": filename
                        }
                    }
                    
                    documents.append(document)
                    
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
                    
        return documents
    
    def _process_pdfs(self, pdf_path: str) -> List:
        """Process PDFs and extract text and images."""
        documents = []
        
        for filename in os.listdir(pdf_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(pdf_path, filename)
                
                try:
                    # Open PDF
                    pdf_document = fitz.open(file_path)
                    
                    # Process each page
                    for page_num, page in enumerate(pdf_document):
                        # Extract text
                        text = page.get_text()
                        
                        # Create document for text
                        if text.strip():
                            document = {
                                "page_content": text,
                                "metadata": {
                                    "source": file_path,
                                    "type": "pdf",
                                    "page": page_num + 1,
                                    "filename": filename
                                }
                            }
                            documents.append(document)
                        
                        # Extract images
                        image_list = page.get_images(full=True)
                        for img_index, img in enumerate(image_list):
                            xref = img[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Create image in memory
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Extract text using OCR
                            extracted_text = pytesseract.image_to_string(image)
                            
                            # Convert image to base64
                            buffered = io.BytesIO()
                            image.save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            
                            # Create document for image
                            document = {
                                "page_content": extracted_text if extracted_text.strip() else f"[PDF Image on page {page_num+1}]",
                                "metadata": {
                                    "source": file_path,
                                    "type": "pdf_image",
                                    "page": page_num + 1,
                                    "image_index": img_index,
                                    "image_data": img_str,
                                    "filename": filename
                                }
                            }
                            documents.append(document)
                            
                except Exception as e:
                    print(f"Error processing PDF {filename}: {str(e)}")
                    
        return documents
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Process a query and retrieve from multimodal sources."""
        if not self.vector_store:
            raise ValueError("Documents must be indexed before querying")
        
        # Perform vector search
        results = self.vector_store.similarity_search(question, k=top_k)
        
        # Separate results by type
        text_results = []
        image_results = []
        pdf_results = []
        
        for doc in results:
            if doc.metadata.get("type") == "image" or doc.metadata.get("type") == "pdf_image":
                image_results.append(doc)
            elif doc.metadata.get("type") == "pdf":
                pdf_results.append(doc)
            else:
                text_results.append(doc)
        
        # Prepare context from all results
        context_parts = []
        
        # Add text documents
        if text_results:
            text_context = "\n\n".join([f"TEXT DOCUMENT {i+1}:\n{doc.page_content}" 
                                      for i, doc in enumerate(text_results)])
            context_parts.append(text_context)
        
        # Add PDF text
        if pdf_results:
            pdf_context = "\n\n".join([f"PDF DOCUMENT {doc.metadata.get('filename')} (Page {doc.metadata.get('page')}):\n{doc.page_content}" 
                                     for doc in pdf_results])
            context_parts.append(pdf_context)
        
        # Add image descriptions
        if image_results:
            image_context = "\n\n".join([f"IMAGE {i+1} ({doc.metadata.get('filename', 'Unknown')}):\n{doc.page_content}" 
                                       for i, doc in enumerate(image_results)])
            context_parts.append(image_context)
        
        # Combine all context
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""
        Answer the question based on the following multimodal context, which includes 
        text documents, PDF content, and image descriptions:
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a comprehensive answer that incorporates information from all relevant sources.
        If the answer includes information from images or PDFs, mention this in your response.
        
        Answer:
        """
        
        answer = self.llm(prompt)
        
        return {
            "answer": answer,
            "source_documents": results
        }
    
    def _load_documents(self, path: str):
        """Load text documents from the specified path."""
        # Implement document loading logic here
        return []

# Example usage
if __name__ == "__main__":
    rag = MultimodalRAG(
        documents_path="./data/text",
        image_path="./data/images",
        pdf_path="./data/pdfs"
    )
    rag.index_documents()
    
    result = rag.query("Explain the architecture shown in the diagrams")
    print(result["answer"])
