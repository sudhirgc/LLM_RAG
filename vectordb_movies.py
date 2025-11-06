import torch
import pandas as pd 
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

df = pd.read_csv("movies.csv")
print(df.info())

df = df.dropna()
df = df.drop_duplicates(subset='id')

df['id'] = df['id'].astype(str)

df['verbiage'] = df['title'] + " : " + df['overview']
print(df.head())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device - {device}")

embedding_fn = SentenceTransformerEmbeddingFunction(device=str(device))

client = chromadb.PersistentClient()
movies_db = client.get_or_create_collection("movies", embedding_function=embedding_fn )

id_list = df['id'].tolist()
verbiage_list = df['verbiage'].tolist()
max_batch_size = 1000
for i in range(0, len(id_list), max_batch_size):
    movies_db.add(ids=id_list[i:i+max_batch_size], documents=verbiage_list[i:i+max_batch_size])

results = movies_db.query(query_texts=['I want to see romantic comedy'],n_results=5)
print(results['documents'])
