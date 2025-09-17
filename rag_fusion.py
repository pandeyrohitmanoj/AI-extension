from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


def create_simple_rag_fusion(documents):
    """
    Input: document collection
    Output: RAG fusion using MultiQueryRetriever
    Purpose: Simpler RAG fusion implementation with built-in LangChain retriever
    """
    # Create base retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    base_retriever = vectorstore.as_retriever()
    
    # Create multi-query retriever (handles query generation automatically)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    return multi_query_retriever

# Usage with MultiQueryRetriever
def simple_fusion_qa(question, documents):
    retriever = create_simple_rag_fusion(documents)
    
    # MultiQueryRetriever automatically generates queries and fuses results
    docs = retriever.get_relevant_documents(question)
    
    # Create answer using retrieved docs
    rag_prompt = PromptTemplate.from_template(
        """Answer based on context: {context}
        Question: {question}"""
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    chain = rag_prompt | llm | StrOutputParser()
    
    context = "\n".join([doc.page_content for doc in docs])
    answer = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": answer,
        "source_documents": docs
    }