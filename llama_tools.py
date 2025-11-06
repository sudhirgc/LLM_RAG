import torch
import pandas as pd
import yfinance
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import transformers
from transformers import AutoTokenizer
from transformers.utils import get_json_schema
import json
import ast

def get_stock_performance(ticker:str):
    """
    This method gets the stock performance for the specified stock ticker as percentage
    Args:
        ticker: The stock ticker for which performance needs to be determined
    """
    print(f'Input received in get_stock_performance, Ticker -> {ticker}')
    df = yfinance.download(tickers=str(ticker),period='ytd')
    start_val = df['Close'].iloc[0].values[0]
    end_val = df['Close'].iloc[-1].values[0]
    print( start_val, end_val)
    perc_change = (end_val - start_val)*100/start_val 
    print(f" Change in Percentage - {perc_change}")
    return perc_change

def execute_pipeline(tokenizer, pipeline, messages):
    tokenizer.apply_chat_template(messages, tools=[get_stock_performance] , tokenize=False, add_generation_prompt=True)
    sequences = pipeline(messages)
    print('************** LLAMA **********************')
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    return sequences

#perc_change = get_stock_performance('NVDA')

function_definitions = get_json_schema(get_stock_performance)

model_name = 'meta-llama/Llama-3.2-1B-Instruct'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device - {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline = transformers.pipeline(task="text-generation", model=model_name,
                                 torch_dtype=torch.float16, device_map="auto",)

system_prompt = """You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. The params to the function needs to be a stock tracker
Call the matching function and return the result. You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n\n{functions}\n""".format(functions=function_definitions)


chat = [
        {"role": "system","content": system_prompt} , 
        {"role": "user", "content": "get me stock performance of company with highest market capitalization"}
    ]

sequences = execute_pipeline(tokenizer, pipeline, chat)
print(sequences[0]['generated_text'][-1]['content'])
print(type(sequences[0]['generated_text'][-1]['content']))
user_response_dict = ast.literal_eval(sequences[0]['generated_text'][-1]['content'])
if (user_response_dict['type'] == 'function'):
    # Get function map 

    # There are 2 cases here, 


    func_dict = user_response_dict['function']['parameters']['ticker']
    func_name = user_response_dict['function']['name']

    #Get method object 
    func_obj = globals()[func_name]
    value = func_obj(func_dict)

    print(value)




""" chat = sequences[0]['generated_text']
chat.append({'role':'user', 'content':'The Ticker is AMD'})

sequences = execute_pipeline(tokenizer, pipeline, chat) """

