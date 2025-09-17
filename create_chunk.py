import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict


CHUNK_SIZES = {
    'default': 1000,
    'function': 800,
    'class': 1400,
    'imports': 500,
    'comments': 600,
    'config': 400  # env, yaml files
}

def extract_file_content(file_path):
   with open(file_path, 'r', encoding='utf-8') as file:
       return file.read()
   
   
def parse_js_code(code: str, extension: str) -> List[Dict]:
   """
   Input: code (string), extension (js/jsx/ts/tsx), chunk_size (number)
   Output: List of semantic chunks with metadata
   Purpose: Parse JS/TS code into semantic chunks respecting size limits
   """
   try:
       # Call Node.js parser
       ast_result = _call_node_parser(code, extension)
       
       # Extract and chunk semantic units
       chunks = _extract_and_chunk(ast_result, code)
       
       return chunks
       
   except Exception as e:
       # Fallback to text chunking if parsing fails
       return _fallback_chunking(code, extension)


def _call_node_parser(code: str, extension: str) -> dict:
   """Bridge to Node.js parser"""
   with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
       request = {'code': code, 'extension': extension}
       json.dump(request, tmp)
       tmp_path = tmp.name
   
   try:
       parser_script = Path(__file__).parent / 'js_parser.js'
       result = subprocess.run(
           ['node', str(parser_script), tmp_path],
           capture_output=True, text=True, timeout=30
       )
       
       if result.returncode != 0:
           raise Exception(f"Parser error: {result.stderr}")
       
       with open(tmp_path + '.result', 'r') as f:
           return json.load(f)
           
   finally:
       os.unlink(tmp_path)
       if os.path.exists(tmp_path + '.result'):
           os.unlink(tmp_path + '.result')

def _extract_and_chunk(ast_data: dict, code: str) -> List[Dict]:
   """Extract semantic units and chunk them by size"""
   chunks = []
   
   # Process imports
   if ast_data.get('imports'):
       import_chunks = _chunk_by_size(ast_data['imports'], code, CHUNK_SIZES['imports'], 'import')
       chunks.extend(import_chunks)
   
   # Process functions
   for func in ast_data.get('functions', []):
       func_chunks = _chunk_single_unit(func, code, CHUNK_SIZES['function'], 'function')
       chunks.extend(func_chunks)
   
   # Process classes
   for cls in ast_data.get('classes', []):
       class_chunks = _chunk_single_unit(cls, code, CHUNK_SIZES['class'], 'class')
       chunks.extend(class_chunks)
   
   # Process exports
   for exp in ast_data.get('exports', []):
       export_chunks = _chunk_single_unit(exp, code, CHUNK_SIZES['imports'], 'export')
       chunks.extend(export_chunks)
   
   return chunks

def _chunk_single_unit(unit: dict, code: str, chunk_size: int, unit_type: str) -> List[Dict]:
   """Chunk a single semantic unit (function/class/export)"""
   start, end = unit['range']
   unit_code = code[start:end]
   name = unit.get('name', 'anonymous')
   
   if len(unit_code) <= chunk_size:
       return [{
           'content': unit_code,
           'metadata': {
               'type': unit_type,
               'name': name,
               'line': unit['line'],
               'size': len(unit_code)
           }
       }]
   else:
       return _split_by_lines(unit_code, chunk_size, name, unit_type, unit['line'])

def _chunk_by_size(units: List[dict], code: str, chunk_size: int, unit_type: str) -> List[Dict]:
   """Group multiple units (like imports) into size-limited chunks"""
   chunks = []
   current_batch = []
   current_size = 0
   
   for unit in units:
       start, end = unit['range']
       unit_code = code[start:end]
       unit_size = len(unit_code)
       
       if unit_size > chunk_size:
           # Finish current batch
           if current_batch:
               chunks.append({
                   'content': '\n'.join(current_batch),
                   'metadata': {'type': f'{unit_type}s', 'count': len(current_batch)}
               })
               current_batch = []
               current_size = 0
           
           # Split large unit
           large_chunks = _split_by_chars(unit_code, chunk_size, f"large_{unit_type}", unit['line'])
           chunks.extend(large_chunks)
       else:
           if current_size + unit_size + 1 > chunk_size and current_batch:
               chunks.append({
                   'content': '\n'.join(current_batch),
                   'metadata': {'type': f'{unit_type}s', 'count': len(current_batch)}
               })
               current_batch = []
               current_size = 0
           
           current_batch.append(unit_code)
           current_size += unit_size + 1
   
   if current_batch:
       chunks.append({
           'content': '\n'.join(current_batch),
           'metadata': {'type': f'{unit_type}s', 'count': len(current_batch)}
       })
   
   return chunks

def _split_by_lines(code: str, chunk_size: int, name: str, unit_type: str, start_line: int) -> List[Dict]:
   """Split code by lines respecting chunk_size"""
   lines = code.split('\n')
   chunks = []
   current_chunk = []
   current_size = 0
   part = 1
   
   for line in lines:
       line_size = len(line) + 1
       if current_size + line_size > chunk_size and current_chunk:
           chunks.append({
               'content': '\n'.join(current_chunk),
               'metadata': {
                   'type': f'{unit_type}_part',
                   'name': f'{name}_part_{part}',
                   'original_name': name,
                   'part': part,
                   'line': start_line
               }
           })
           current_chunk = []
           current_size = 0
           part += 1
       
       current_chunk.append(line)
       current_size += line_size
   
   if current_chunk:
       chunks.append({
           'content': '\n'.join(current_chunk),
           'metadata': {
               'type': f'{unit_type}_part',
               'name': f'{name}_part_{part}',
               'original_name': name,
               'part': part,
               'line': start_line
           }
       })
   
   # Update total_parts
   for chunk in chunks:
       chunk['metadata']['total_parts'] = part
   
   return chunks

def _split_by_chars(code: str, chunk_size: int, name: str, start_line: int) -> List[Dict]:
   """Split by character count when line splitting isn't suitable"""
   chunks = []
   for i in range(0, len(code), chunk_size):
       chunk_code = code[i:i + chunk_size]
       chunks.append({
           'content': chunk_code,
           'metadata': {
               'type': 'text_part',
               'name': f'{name}_char_{i // chunk_size}',
               'line': start_line
           }
       })
   return chunks

def _fallback_chunking(code: str, extension: str) -> List[Dict]:
   """Simple text chunking when parsing fails"""
   chunks = []
   chunk_size = CHUNK_SIZES['default']
   for i in range(0, len(code), chunk_size):
       chunk = code[i:i + chunk_size]
       chunks.append({
           'content': chunk,
           'metadata': {
               'type': 'text_chunk',
               'name': f'chunk_{i // chunk_size}',
               'extension': extension
           }
       })
   return chunks



# jsContent = extract_file_content('/home/monkey/Downloads/ml-apps/LAST-APP/test/third.ts')
# chunks = parse_js_code(jsContent,'tsx')
# print(chunks)