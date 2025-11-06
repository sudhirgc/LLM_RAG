import torch
import pandas as pd 
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

pdf_reader = PdfReader("p17.pdf")
irs_texts = [page.extract_text().strip() for page in pdf_reader.pages if (len(page.extract_text().strip()) > 0)]
print(f'No of IRS text {len(irs_texts)} ')
print(irs_texts[0])

recursive_splitter = RecursiveCharacterTextSplitter(
                        separators= ["\n\n", "\n", ". ", " ", ""],
                        chunk_size=1000,
                        chunk_overlap=0.2)

char_split_texts = []
for irs_text in irs_texts:
    char_split_texts.extend( recursive_splitter.split_text(irs_text) ) 

print(f'No of Char Splitted text {len(char_split_texts)} ')

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=10,
    tokens_per_chunk=256
    )

token_split_texts = []
for char_txt in char_split_texts:
    token_split_texts.extend(token_splitter.split_text(char_txt))

print(f'No of Token Splitted text {len(token_split_texts)} ')

print(f'Token Splitted text {token_split_texts[0]} ')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device - {device}")

embedding_fn = SentenceTransformerEmbeddingFunction(device=str(device))
client = chromadb.PersistentClient()
irs_db = client.get_or_create_collection("IRS_Income_Tax", embedding_function=embedding_fn )

id_list = [str(i) for i in range(len(token_split_texts))]

max_batch_size = 1000

for i in range(0, len(id_list), max_batch_size):
    irs_db.add(
        ids=id_list, documents=token_split_texts
        )

results = irs_db.query(query_texts=['What is the standard deduction for married people'],n_results=5)
print(results['documents'])

