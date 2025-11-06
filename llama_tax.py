import torch
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import transformers
from transformers import AutoTokenizer

model_name = 'meta-llama/Llama-3.2-1B-Instruct'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device - {device}")
embedding_fn = SentenceTransformerEmbeddingFunction(device=str(device))
client = chromadb.PersistentClient()
irs_db = client.get_or_create_collection("IRS_Income_Tax", embedding_function=embedding_fn )

query = 'What is the tax rate for unemployed person'
results = irs_db.query(query_texts=[query],n_results=10)
print(results['documents'])

docs = results["documents"][0]
joined_information = ';'.join([f'{doc}' for doc in docs])

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline = transformers.pipeline(task="text-generation", model=model_name,
                                 torch_dtype=torch.float16, device_map="auto",)
chat = [
        {
            "role": "system","content": "You will be shown the user's question, and the relevant information. Answer the user's question and summarize using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {joined_information}"}
    ]

def execute_pipeline(tokenizer, pipeline, messages):
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sequences = pipeline(messages)
    print('************** LLAMA **********************')
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    return sequences

sequences = execute_pipeline(tokenizer, pipeline, chat)

chat = sequences[0]['generated_text']
chat.append({'role':'user', 'content':' what is tax rate for income of 11,600?'})

sequences = execute_pipeline(tokenizer, pipeline, chat)