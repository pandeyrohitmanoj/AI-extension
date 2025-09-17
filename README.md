Setup:

To create environment: 
conda env create -f environment.yml

To start the Fast API :
python main.py



Endpoints:
1. Summarize-file: curl -X Post http://localhost:8000/summarize-file -H "Content-Type:application/json" -d '{query:"code string", root_address:"directory at which yuou want to save index for codebase', target_address:"Directory of your codebase"'}'
2. optimize-file: curl -X Post http://localhost:8000/summarize-file -H "Content-Type:application/json" -d '{query:"code string", root_address:"directory at which yuou want to save index for codebase', target_address:"Directory of your codebase"'}'