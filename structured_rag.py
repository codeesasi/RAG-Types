from typing import List, Dict, Any, Union
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import sqlite3
import pandas as pd
import networkx as nx

class StructuredRAG:
    """
    Structured RAG implementation that works with structured data sources
    (databases, knowledge graphs, APIs) rather than just unstructured text.
    """
    
    def __init__(self, documents_path: str = None, db_path: str = None, kg_path: str = None):
        self.embedding_model = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.3)
        self.documents_path = documents_path
        self.db_path = db_path
        self.kg_path = kg_path
        
        # Initialize vector store for unstructured documents (optional)
        self.vector_store = None
        
        # Initialize database connection (if provided)
        self.db_conn = None
        if db_path:
            try:
                self.db_conn = sqlite3.connect(db_path)
                print(f"Connected to database: {db_path}")
            except Exception as e:
                print(f"Error connecting to database: {str(e)}")
        
        # Initialize knowledge graph (if provided)
        self.knowledge_graph = None
        if kg_path:
            try:
                # This is a placeholder - in a real implementation, you would load your KG
                # using appropriate libraries (e.g., RDFLib for RDF graphs, NetworkX for general graphs)
                self.knowledge_graph = nx.DiGraph()
                print(f"Loaded knowledge graph from: {kg_path}")
            except Exception as e:
                print(f"Error loading knowledge graph: {str(e)}")
    
    def index_documents(self):
        """Create vector embeddings for unstructured documents (if any)."""
        if not self.documents_path:
            return 0
            
        documents = self._load_documents(self.documents_path)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        return len(documents)
    
    def get_schema(self) -> Dict:
        """
        Extract schema information from the connected database.
        
        Returns:
            Dictionary with table and column information
        """
        if not self.db_conn:
            return {"error": "No database connection"}
        
        tables = {}
        cursor = self.db_conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in cursor.fetchall()]
        
        # Get schema for each table
        for table in table_names:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            tables[table] = {
                "columns": [{"name": col[1], "type": col[2]} for col in columns],
                "sample_data": self._get_sample_data(table)
            }
        
        return tables
    
    def _get_sample_data(self, table: str, limit: int = 3) -> List[Dict]:
        """Get sample data from a database table."""
        if not self.db_conn:
            return []
            
        try:
            query = f"SELECT * FROM {table} LIMIT {limit};"
            df = pd.read_sql_query(query, self.db_conn)
            return df.to_dict(orient='records')
        except Exception as e:
            print(f"Error getting sample data for {table}: {str(e)}")
            return []
    
    def query_to_sql(self, question: str) -> str:
        """
        Convert a natural language question to SQL.
        
        Args:
            question: The natural language question
            
        Returns:
            SQL query string
        """
        if not self.db_conn:
            return "ERROR: No database connection"
        
        # Get database schema
        schema = self.get_schema()
        schema_str = json.dumps(schema, indent=2)
        
        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
            You are an expert in converting natural language questions to SQL queries.
            
            Below is the database schema:
            {schema}
            
            Please convert the following question to a SQL query that can be executed against this database.
            Only return the SQL query, nothing else.
            
            Question: {question}
            
            SQL:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        sql_query = chain.run(schema=schema_str, question=question)
        
        return sql_query.strip()
    
    def execute_sql(self, sql_query: str) -> Dict:
        """
        Execute SQL query and return results.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Dictionary with query results
        """
        if not self.db_conn:
            return {"error": "No database connection"}
        
        try:
            # Check if query is read-only (for safety)
            if any(keyword in sql_query.lower() for keyword in ["insert", "update", "delete", "drop", "alter", "create"]):
                return {"error": "Only SELECT queries are allowed"}
            
            # Execute query
            df = pd.read_sql_query(sql_query, self.db_conn)
            
            # Convert to records
            records = df.to_dict(orient='records')
            
            return {
                "sql": sql_query,
                "columns": df.columns.tolist(),
                "rows": len(records),
                "data": records
            }
        except Exception as e:
            return {"error": str(e), "sql": sql_query}
    
    def query_knowledge_graph(self, question: str) -> Dict:
        """
        Query the knowledge graph based on natural language question.
        
        Args:
            question: The natural language question
            
        Returns:
            Dictionary with query results
        """
        if not self.knowledge_graph:
            return {"error": "No knowledge graph available"}
        
        # This is a simplified implementation
        # In a real system, you would:
        # 1. Parse the question to identify entities and relations
        # 2. Map to your KG schema
        # 3. Execute graph queries (e.g., SPARQL for RDF graphs)
        
        # For demonstration purposes, we'll use a simple approach
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            I want to query a knowledge graph based on this question: {question}
            
            What would be:
            1. The main entity/entities to search for
            2. The relationships or properties to retrieve
            3. Any constraints or filters
            
            Format your answer as JSON with keys "entities", "relationships", and "filters".
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        kg_query_str = chain.run(question=question)
        
        # Parse the result (in a real implementation, handle JSON parsing errors)
        try:
            kg_query = json.loads(kg_query_str)
        except:
            kg_query = {
                "entities": ["unknown"],
                "relationships": ["unknown"],
                "filters": []
            }
        
        # Placeholder for KG query results
        # In a real implementation, this would execute actual graph queries
        results = {
            "query": kg_query,
            "results": [
                {"note": "This is a placeholder for actual knowledge graph results"}
            ]
        }
        
        return results
    
    def query(self, question: str) -> Dict:
        """
        Process a query across structured and unstructured data sources.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer and sources
        """
        # Step 1: Determine which data sources to query
        source_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            Given the following question, which data sources would be most appropriate to query?
            
            Question: {question}
            
            Available sources:
            - Unstructured documents (for general knowledge and text information)
            - SQL Database (for structured, tabular data)
            - Knowledge Graph (for entities and their relationships)
            
            Return a JSON object with keys "use_documents", "use_database", and "use_knowledge_graph", 
            each with a boolean value (true/false).
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=source_prompt)
        source_result = chain.run(question=question)
        
        # Parse the source selection (handle potential JSON parsing errors)
        try:
            sources = json.loads(source_result)
        except:
            # Default to using all available sources
            sources = {
                "use_documents": self.vector_store is not None,
                "use_database": self.db_conn is not None,
                "use_knowledge_graph": self.knowledge_graph is not None
            }
        
        # Step 2: Query each selected data source
        results = {}
        context_parts = []
        
        # Query unstructured documents if selected and available
        if sources.get("use_documents", False) and self.vector_store:
            docs = self.vector_store.similarity_search(question, k=3)
            doc_texts = [doc.page_content for doc in docs]
            results["documents"] = docs
            context_parts.append("DOCUMENT INFORMATION:\n" + "\n\n".join(doc_texts))
        
        # Query database if selected and available
        if sources.get("use_database", False) and self.db_conn:
            sql_query = self.query_to_sql(question)
            sql_results = self.execute_sql(sql_query)
            results["database"] = sql_results
            
            if "error" not in sql_results:
                # Format SQL results as readable text
                table_data = pd.DataFrame(sql_results["data"])
                table_str = table_data.to_string(index=False)
                context_parts.append(f"DATABASE INFORMATION:\nSQL Query: {sql_query}\n\nResults:\n{table_str}")
            else:
                context_parts.append(f"DATABASE ERROR: {sql_results['error']}")
        
        # Query knowledge graph if selected and available
        if sources.get("use_knowledge_graph", False) and self.knowledge_graph:
            kg_results = self.query_knowledge_graph(question)
            results["knowledge_graph"] = kg_results
            context_parts.append("KNOWLEDGE GRAPH INFORMATION:\n" + json.dumps(kg_results, indent=2))
        
        # Step 3: Combine information and generate answer
        if not context_parts:
            return {"answer": "No data sources were available or selected for this query."}
        
        context = "\n\n" + "-"*50 + "\n\n".join(context_parts)
        
        answer_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Answer the following question based on the provided information from multiple data sources.
            
            Question: {question}
            
            Information from data sources:
            {context}
            
            Provide a comprehensive answer that synthesizes information from all relevant sources.
            If data comes from a specific source (database, knowledge graph), mention this in your answer.
            If the information contains numerical data or statistics, include these in your answer.
            
            Answer:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=answer_prompt)
        answer = chain.run(context=context, question=question)
        
        return {
            "answer": answer.strip(),
            "sources_used": sources,
            "raw_results": results
        }
    
    def _load_documents(self, path: str):
        """Load documents from the specified path."""
        # Implement document loading logic here
        # This would typically use langchain document loaders
        pass

# Example usage
if __name__ == "__main__":
    # Initialize with paths to different data sources
    rag = StructuredRAG(
        documents_path="./data/documents",
        db_path="./data/sample.db",
        kg_path="./data/knowledge_graph.json"
    )
    
    # Index unstructured documents (if any)
    rag.index_documents()
    
    # Example queries that would leverage different data sources
    db_question = "What are the top 5 products by sales in Q1 2023?"
    kg_question = "What are the relationships between diabetes and cardiovascular diseases?"
    hybrid_question = "How has the company's market share evolved compared to competitors, and what factors contributed to this trend?"
    
    # Execute a query that might use multiple data sources
    result = rag.query(hybrid_question)
    
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES USED:")
    for source, used in result["sources_used"].items():
        print(f"- {source}: {'Yes' if used else 'No'}")
