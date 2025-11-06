# LLM_RAG
This Repository contains code demonstrating use of RAG with VectorDBs, LLMs, Tools. There are 2 main functionalities implemented here.
- A chatbot (incomplete as of now) which tax answers question based on publication 17. The p17 document is read and pushed to chromaDB by vectordb_irs.py file. Once the data is in chromaDB, llama_tax.py has the sample code to query the DB and then submit the results to llama model for RAG consolidation. 
- Another chatbot which suggests buy/sell for a given stock based on Technical Analysis. This code is implememted in llama_stocks.py  
The code in this repository need to run in a venv or conda environment for which requirements.txt is provided. Create a venv or conda environment with python==3.12. Latest versions of python are creating numpy issues and breaking the envoironment. For best results create the environment and get the torch install commands from pytorch website. 
