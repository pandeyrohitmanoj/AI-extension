
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from create_vector import save_directory, load_or_create_vectordb, save_file_to_vector
from create_rag import code_summarization_rag_langchain, code_optimization_rag_langchain, debug_vectordb_first
# Create FastAPI instance
app = FastAPI(title="My API", description="A simple API with GET and POST endpoints", version="1.0.0")

class create_vector(BaseModel):
    root_address: str
    target_address: str  # Add this field
    query: str

# Root endpoint
vector_db_list:Dict[str,Dict] = {}


def clean_rag_result(result):
    """More aggressive SWIG object cleaning"""
    def deep_clean(obj):
        if obj is None:
            return None
        
        # Check for SWIG objects
        if 'SwigPyObject' in str(type(obj)) or 'Swig' in str(type(obj)):
            try:
                # Try different conversion methods
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, '__len__') and hasattr(obj, '__getitem__'):
                    return [deep_clean(obj[i]) for i in range(len(obj))]
                elif hasattr(obj, '__iter__'):
                    return [deep_clean(item) for item in obj]
                else:
                    return str(obj)  # Convert to string as last resort
            except Exception as e:
                print(f"Error converting SWIG object: {e}")
                return f"<SWIG_OBJECT:{type(obj).__name__}>"
        
        # Handle standard Python types
        elif isinstance(obj, dict):
            return {k: deep_clean(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [deep_clean(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # For any other unknown type, convert to string
            try:
                return str(obj)
            except:
                return f"<UNKNOWN_TYPE:{type(obj).__name__}>"
    
    return deep_clean(result)



# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI!", "status": "running"}

def get_vector_db(address: str, target_address: str) -> Dict:
    vector_db: Dict = {}  # Initialize local variable too
    
    # Check if address exists in dict and has a value
    if address in vector_db_list and vector_db_list[address]:
        vector_db = vector_db_list[address]
    else:
        # Create new vector db if not exists
        vector_db = debug_vectordb_first(address)
        if not vector_db['exist']:
            save_directory(target_address, vector_db,address)
        vector_db_list[address] = vector_db
    
    return vector_db
    
@app.post("/summarize-file")
def summarize_file(body: create_vector):
    try:
        vector_db:Dict= get_vector_db(body.root_address,body.target_address)
        result = code_summarization_rag_langchain(body.query,body.root_address, vector_db)
        result = clean_rag_result(result)
        return {"result":result,"error": False}
    except:
        print("bad error")
        return {"error":True}
        

@app.post("/optimize-file", response_model=None)
def optimize_file(body: create_vector):
    try:
        vector_db:Dict= get_vector_db(body.root_address,body.target_address)
        result = code_optimization_rag_langchain(body.query,body.root_address,vector_db)
        result = clean_rag_result(result) 
        return {"result": result, "error": False}
    except:
        print("bad error happened")
        return {"error":True}
    
if __name__ == "__main__":
    print('app is running')
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 