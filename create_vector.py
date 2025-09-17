
import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional

from create_chunk import extract_file_content,parse_js_code

def save_vectordb(address: str, vectordb: Dict) -> bool:
   """
   Input: address (file path), vectordb (dict with index, metadata, encoder)
   Output: boolean success status
   Purpose: Save vector database to disk
   """
   try:
       address_path = Path(address) / 'vector_storage'
       address_path.mkdir(parents=True, exist_ok=True)
       
       # Save FAISS index
       index_path = address_path / "vector.index"
       faiss.write_index(vectordb['index'], str(index_path))
       
       # Save metadata
       metadata_path = address_path / "metadata.pkl"
       with open(metadata_path, 'wb') as f:
           pickle.dump(vectordb['metadata'], f)
       
       # Save encoder info (model name for reloading)
       encoder_path = address_path / "encoder.pkl"
       encoder_info = {
           'model_name': vectordb['encoder'].get_sentence_embedding_dimension(),
           'dimension': vectordb['encoder'].get_sentence_embedding_dimension()
       }
       with open(encoder_path, 'wb') as f:
           pickle.dump(encoder_info, f)
       
       return True
       
   except Exception as e:
       print(f"Error saving vectordb: {e}")
       return False

# address is the parent directory for the file, and eact directory for the folder
def load_or_create_vectordb(address: str) -> Dict:
   """
   Input: address (file path)
   Output: vectordb dictionary with index, metadata, encoder
   Purpose: Load existing vectordb or create new one if not exists
   """
   
   address_path = Path(address)
   if not address_path.exists():
       return {"exist": False}
    
   if not address_path.is_dir():
       address_path = address_path.parent / "vector_storage"
   else:
       address_path = address_path / "vector_storage"
       
   index_path = address_path / "vector.index"
   metadata_path = address_path / "metadata.pkl"
   
   # Check if vectordb exists
   if index_path.exists() and metadata_path.exists():
       try:
           # Load existing vectordb
           index = faiss.read_index(str(index_path))
           
           with open(metadata_path, 'rb') as f:
               metadata = pickle.load(f)
           
           # Initialize encoder (same model as used during creation)
           encoder = SentenceTransformer('all-MiniLM-L6-v2')
           
           return {
               'index': index,
               'metadata': metadata,
               'encoder': encoder,
               'address': str(address_path),
               "exist": True,
           }
           
       except Exception as e:
           print(f"Error loading vectordb, creating new: {e}")
   
   # Create new vectordb
   address_path.mkdir(parents=True, exist_ok=True)
   encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
   index = faiss.IndexFlatIP(384)  # Inner Product similarity
   
   return {
       'index': index,
       'metadata': [],
       'encoder': encoder,
       'address': str(address_path),
        "exist": False,
   }

# def add_chunk_to_vectordb(chunks: List[Dict], address: str, vectordb: Dict) :
#    """
#    Input: chunks (list of dicts with content/metadata), address (file path), vectordb (optional existing db)
#    Output: boolean success status
#    Purpose: Add list of chunks to vectordb and save to disk
#    """
#    try:
#        if not chunks:
#            return {}
       
#        # Filter valid chunks and extract contents
#        valid_chunks = []
#        contents = []
       
#        for chunk in chunks:
#            content = chunk.get('content', '')
#            if content.strip():  # Only add non-empty chunks
#                valid_chunks.append(chunk)
#                contents.append(content)
       
#        if not valid_chunks:
#            return {}
       
#        # Generate embeddings for all chunks in batch
#        embeddings = vectordb['encoder'].encode(contents)
#        embeddings = embeddings.astype('float32')
       
#        # Add all embeddings to index at once
#        vectordb['index'].add(embeddings)
       
#        # Add metadata for each chunk
#        start_vector_id = vectordb['index'].ntotal - len(valid_chunks)
       
#        for i, chunk in enumerate(valid_chunks):
#            metadata = chunk.get('metadata', {})
#            metadata['content_length'] = len(contents[i])
#            metadata['vector_id'] = start_vector_id + i
           
#            vectordb['metadata'].append(metadata)
        
#    except Exception as e:
#        print(f"Error adding chunks to vectordb: {e}")
#        return {}
   

def add_chunk_to_vectordb(chunks: List[Dict], address: str, vectordb: Dict) -> Dict:
    """
    Input: chunks (list of dicts with content/metadata), address (file path), vectordb (optional existing db)
    Output: boolean success status
    Purpose: Add list of chunks to vectordb and save to disk
    """
    try:
        # Load vectordb if not provided
        
        if not chunks:
            return {}
        
        # Filter valid chunks and extract contents
        valid_chunks = []
        contents = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            if content.strip():  # Only add non-empty chunks
                valid_chunks.append(chunk)
                contents.append(content)
        
        if not valid_chunks:
            return {}
        
        # Generate embeddings for all chunks in batch
        embeddings = vectordb['encoder'].encode(contents)
        embeddings = embeddings.astype('float32')
        
        # Add all embeddings to index at once
        vectordb['index'].add(embeddings)
        
        # Add metadata for each chunk - INCLUDE CONTENT!
        start_vector_id = vectordb['index'].ntotal - len(valid_chunks)
        
        for i, chunk in enumerate(valid_chunks):
            metadata = chunk.get('metadata', {})
            metadata['content_length'] = len(contents[i])
            metadata['vector_id'] = start_vector_id + i
            # THIS IS THE FIX - STORE CONTENT IN METADATA
            metadata['content'] = contents[i]  # <-- ADD THIS LINE
            
            vectordb['metadata'].append(metadata)
        
        # Create address path and save updated vectordb
        return vectordb
        
    except Exception as e:
        print(f"Error adding chunks to vectordb: {e}")
        return {} 

# Helper function to get vectordb stats
def get_vectordb_stats(address: str) -> Dict:
   """
   Input: address (file path)
   Output: stats dictionary
   Purpose: Get information about vectordb size and contents
   """
   try:
       vectordb = load_or_create_vectordb(address)
       
       return {
           'total_vectors': vectordb['index'].ntotal,
           'dimension': vectordb['index'].d,
           'metadata_count': len(vectordb['metadata']),
           'index_size_mb': os.path.getsize(Path(address) / "vector.index") / (1024 * 1024) if (Path(address) / "vector.index").exists() else 0
       }
       
   except Exception as e:
       return {'error': str(e)}
   

from pathlib import Path
allowed_extension = ['js','ts','jsx','tsx']

def save_file_to_vector(pathValue: str,vectorDb:Dict, vector_db_address: str):
    extension = pathValue.split('.')[-1]
    
    if allowed_extension.count(extension)==0:
        return
    jsContent = extract_file_content(pathValue)
    print(extension,'jumpy')
    chunks = parse_js_code(jsContent,extension)
    add_chunk_to_vectordb(chunks, pathValue, vectorDb)
   

def save_directory(target_dir: str,vector_db:Dict,vectordb_address):
    path = Path(target_dir)
    for file in path.iterdir():
        if file.is_dir(): 
            save_directory(str(file),vector_db,vectordb_address)
        else: 
            save_file_to_vector(str(file),vector_db,vectordb_address)
        
    save_vectordb(vectordb_address, vector_db)

# vectorDb = save_file_to_vector(pathValue)
# save_file_to_vector(pathValue1, vectorDb)
# print(str(Path(pathValue).parent)+'jumppy')
# save_vectordb(str(Path(pathValue).parent),vectorDb)

# dir_path = '/home/monkey/Downloads/ml-apps/ic-backend/services/user-service'
# target_path='/home/monkey/Downloads/ml-apps/ic-backend/services/user-service/src'
# vector_db = load_or_create_vectordb(dir_path)
# save_directory(target_path,vector_db,dir_path)
# # save_vectordb(dir_path,vector_db)