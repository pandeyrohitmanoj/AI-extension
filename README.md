Code assistant for Vs Code:


Setup:

To create environment: 
conda env create -f environment.yml
npm i

To start the Fast API :
python main.py



Endpoints:
1. Summarize-file: curl -X Post  -H "Content-Type:application/json" -d '{query:"code string", root_address:"directory at which yuou want to save index for codebase', target_address:"Directory of your codebase"'}' http://localhost:8000/summarize-file
2. optimize-file: curl -X Post -H "Content-Type:application/json" -d '{query:"code string", root_address:"directory at which yuou want to save index for codebase', target_address:"Directory of your codebase"'}' http://localhost:8000/summarize-file 