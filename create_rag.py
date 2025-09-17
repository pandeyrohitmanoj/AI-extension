from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.schema import Document, BaseRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from create_vector import load_or_create_vectordb
import json


def create_local_embeddings_langchain():
    class LocalEmbeddings(Embeddings):
        def __init__(self):
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        
        def embed_query(self, text: str) -> List[float]:
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
    
    return LocalEmbeddings()

def debug_vectordb_first(address: str):
    """Add this to check what's in your vectordb"""
    vectordb = load_or_create_vectordb(address)
    
    print(f"Total vectors: {vectordb['index'].ntotal}")
    print(f"Total metadata: {len(vectordb['metadata'])}")
    
    # Check first few entries
    for i, metadata in enumerate(vectordb['metadata'][:3]):
        print(f"\nEntry {i}:")
        print(f"  Type: {metadata.get('type', 'unknown')}")
        print(f"  Name: {metadata.get('name', 'unnamed')}")
        print(f"  Has content: {'content' in metadata}")
        print(f"  Content length: {len(metadata.get('content', ''))}")
        if 'content' in metadata:
            print(f"  Content preview: {metadata['content'][:100]}...")
    return vectordb

def create_langchain_vectorstore(address: str, embeddings):
    vectordb = load_or_create_vectordb(address)
    
    class LocalVectorStore(VectorStore):
        def __init__(self, vectordb_instance, embeddings_instance):
            self.vectordb = vectordb_instance
            self._embeddings = embeddings_instance
        
        def add_texts(self, texts: List[str], metadatas: List[dict] = [], **kwargs) -> List[str]:
            """Required method - implement even if not used"""
            return []
        
        def similarity_search(self, query: str, k: int = 5) -> List[Document]:
            print(f"Searching for: '{query}'")  # Debug
            
            query_embedding = self.vectordb['encoder'].encode([query])
            scores, indices = self.vectordb['index'].search(query_embedding, k)
            
            documents = []
            print(f"Found {len(indices[0])} potential matches")
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.vectordb['metadata']) and idx >= 0:
                    metadata = self.vectordb['metadata'][idx]
                    
                    # Check if content exists
                    content = metadata.get('content', '')
                    if not content:
                        print(f"Warning: No content for index {idx}, metadata keys: {list(metadata.keys())}")
                        continue
                    
                    print(f"Found: {metadata.get('name', 'unnamed')} (score: {score:.4f})")
                    
                    doc = Document(
                        page_content=content,
                        metadata={**metadata, 'similarity_score': float(score)}
                    )
                    documents.append(doc)
            
            print(f"Returning {len(documents)} documents with content")
            return documents
        
        @classmethod
        def from_texts(cls, texts: List[str], embedding: Embeddings, **kwargs):
            pass
        
        @property
        def embeddings(self):
            return self._embeddings
    
    return LocalVectorStore(vectordb, embeddings)

GEMINI_KEY='AIzaSyDjdixm8_feU60KCTGEKhmQfaCWsXOUVtc'

def token_tracking(response):
    if hasattr(response, 'response_metadata'):
        usage = response.response_metadata.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens: {total_tokens}")     
                                                                                                         
def setup_langchain_rag(address: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_KEY,
        temperature=0.3
    )
    
    embeddings = create_local_embeddings_langchain()
    vectorstore = create_langchain_vectorstore(address, embeddings)
    
    # Test vectorstore directly first
    test_docs = vectorstore.similarity_search("functions", k=3)
    print(f"Direct vectorstore test: found {len(test_docs)} documents")
    
    base_retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": 10, "score_threshold": 0.85})
    
    return llm, base_retriever

# def create_rag_chain(llm, retriever):
#     prompt_template = """Use the following code examples to answer the question about code analysis and summarization.

# Context (Code Examples):
# {context}

# Question: {question}

# Provide a clear, structured answer that:
# 1. Summarizes the main functions/components/classes found
# 2. Explains what each code block does
# 3. Identifies key programming patterns or concepts
# 4. Lists function names, input, output and their purposes
# 5. explain syntax of the third party module.

# Answer:"""
    
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )
    
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={"prompt": prompt},
#         return_source_documents=True
#     )
    
#     return qa_chain

def create_rag_chain_for_summarization(llm, retriever):
    
    
    prompt_template = PromptTemplate(
        input_variables=["context","question"],
        template="Process this JSON data: {question}\n, based on the context:{context}"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    
    return qa_chain

def code_summarization_rag_langchain(query: str, address: str, vector_db: Optional[Dict]) -> Dict:
    try:
        # Debug vectordb contents first
        print("=== Debugging VectorDB ===")
        # Setup RAG components
        llm, retriever = setup_langchain_rag(address)
        
        # Test retriever directly
        print(f"\n=== Testing Retriever ===")
        docs = retriever.get_relevant_documents(query)
        print(f"Retriever found {len(docs)} documents")
        
        if not docs:
            return { "error":False, "result":"No relevant documents found. Check if your vectordb has content properly stored."}
        
        # Create RAG chain
        qa_chain = create_rag_chain_for_summarization(llm, retriever)
        request = json.dumps({
            "task":"summarize and optimize the code provided in input key in this JSON, and respond the answer in the response_syntax, and folow the guidelines present in this json",
            "response_syntax":{"result": "function code(param){ // comments"},
            "input": query,
            "guidelines": ['no text other than code', 'no wrapper string like "javascript ``` ```",','response is a simple multi-line string','input is in one block, and i want the answer to be in 1 single block']
        })
        # Run query
        result = qa_chain.invoke(f"Process this JSON:{request}")
        token_tracking(result)
        return { "result": result.content.replace('```javascript','').replace('```',''), "error": False}
        
    except Exception as e:
        return { "error": False, "result": f"Error in RAG pipeline: {e}"}



def create_rag_chain_for_optimization(llm, retriever):
    prompt_template = """
    You are an expert software engineer specializing in code optimization and performance improvements. Your task is to analyze and optimize the provided code while considering the given context.


## Context: {context}
**Project Type:** [e.g., Web API, Data Processing, Machine Learning, etc.]
**Programming Language:** [e.g., Python, JavaScript, TypeScript, etc.]
**Framework/Libraries:** [e.g., FastAPI, React, TensorFlow, etc.]
**Performance Requirements:** [e.g., Handle 1000+ requests/second, Process large datasets, Real-time processing]
**Constraints:** [e.g., Memory limitations, CPU constraints, Must maintain backward compatibility, code repetition]
**Target Environment:** [e.g., Production server, Mobile device, Cloud deployment]
**Current Issues:** [e.g., Slow response times, High memory usage, CPU bottlenecks]

for eg:
Question: optimize the below code ''

## Code to Optimize:{question}"""
   
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def code_optimization_rag_langchain(query: str, address: str,vector_db: Dict) -> Dict:
    try:
        # Setup RAG components
        llm, retriever = setup_langchain_rag(address)
        
        # Test retriever directly
        print(f"\n=== Testing Retriever ===")
        docs = retriever.get_relevant_documents(query)
        print(f"Retriever found {len(docs)} documents")
        
        if not docs:
            return { "error":False, "result":"No relevant documents found. Check if your vectordb has content properly stored."}
        
        # Create RAG chain
        qa_chain = create_rag_chain_for_optimization(llm, retriever)
        
        # Run query
        result = qa_chain.invoke({"query": query})
        
        return { "result": result["result"], "vector_db": vector_db, "error": False}
        
    except Exception as e:
        return { "error": False, "result": f"Error in RAG pipeline: {e}"}

# Test the debug version
# print("=== Starting Debug ===")



# summary = code_summarization_rag_langchain(
#     'optimize the code in fourth.tsx',
#     '/home/monkey/Downloads/ml-apps/LAST-APP/test'
# )

# print("=== Final Result ===")
# print(summary)