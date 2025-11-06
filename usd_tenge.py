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
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator
import chainlit as cl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_tenge_performance(ticker:str='USD'):
    """
    This method gets the stock performance for the specified stock ticker as percentage
    Args:
        ticker: The stock ticker for which performance needs to be determined
    """
    logger.info(f'Input received in get_tenge_performance, Ticker -> {ticker}')

    # Lets  confirm the ticker, convert to ticker if needed......
    try:
        
        df = pd.read_csv("us_tenge.csv")
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
        df.set_index('Date', inplace=True)
        print(df.head())

        print(df.info())

        df['Daily_Diff'] = df['Close'].diff(periods=1)
        df['Weekly_Diff'] = df['Close'].diff(periods=7)

        fig, ax = plt.subplots(figsize=(12, 10))
        #ax.plot(df['Date'].values, df['Close'].values, "b--", label=f"{ticker}")
        ax.plot(df.index.values, df['Daily_Diff'].values, "r--", label=f"Daily Diff {ticker}")
        ax.plot(df.index.values, df['Weekly_Diff'].values, "g--", label=f"Weekly Diff {ticker}")
        ax.legend(loc="best") 

        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()

        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df['Histogram'] = df['MACD'] - df['Signal']

        rsi_indicator = RSIIndicator(close=df['Close'].squeeze())
        df['rsi'] = rsi_indicator.rsi()

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(df.index.values, df['Close'].values, "y--", label=f"{ticker}")
        ax.plot(df.index.values, df['SMA_200'].values, "c--", label="SMA_200")
        ax.plot(df.index.values, df['SMA_50'].values, "m--", label="SMA_50")
        ax.plot(df.index.values, df['rsi'].values, "b--", label="RSI")
        ax.legend(loc="upper right") 
        ax2 = ax.twinx()
        ax2.plot(df.index.values, df['MACD'].values, "r--", label="MACD")
        ax2.plot(df.index.values, df['Signal'].values, "g--", label="Signal")
        ax2.bar(df.index.values, df['Histogram'].values, label="Histogram")
        ax2.legend(loc="upper left") 

        print(df.head(-10))

        last_macd = df['MACD'].iloc[-1]
        last_signal = df['Signal'].iloc[-1]
        last_histogram = df['Histogram'].iloc[-1]
        last_rsi = round(df['rsi'].iloc[-1])

        last_stock_price = df['Close'].iloc[-1]
        last_200_price = df['SMA_200'].iloc[-1]
        last_50_price = df['SMA_50'].iloc[-1]

        decision = 'Hold'
        if last_histogram < 0:
            if last_rsi > 70:
                decision = 'Sell'
            else:
                decision = 'Possible Sell'
        elif last_histogram > 0:
            if last_rsi < 30:
                decision = 'Buy'
            else:
                decision = 'Possible Buy'
        
        
        start_val = df['Close'].iloc[0]
        end_val = df['Close'].iloc[-1]
        print( start_val, end_val)
        perc_change = (end_val - start_val)*100/start_val 
        print(f" Change in Percentage - {perc_change}")

        return ticker, decision, last_histogram, last_macd, last_signal, last_rsi, (last_stock_price - last_200_price), (last_50_price - last_200_price) , perc_change
    except Exception as e:
        logger.error('get_tenge_performance :: Error calculating performance -> ', e)
    return ticker, 'Hold', 0, 0, 0, 0, 0, 0,0


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    # This is the input from the USER -> 
    ticker, decision, histogram, macd, signal, rsi, adx, price_over_200, sma50_over_200 , perc_change = get_tenge_performance(message.content)

    # Send a response back to the user
    await cl.Message(
        content=f"Ticker: {ticker}, Decision: {decision}, MACD Rise/Fall: {histogram:.2f}, RSI: {rsi}, Price/200 {price_over_200}, sma 50 / 200 {sma50_over_200}  , % Change: {perc_change:.2f}",
    ).send()

ticker, decision, last_histogram, last_macd, last_signal, last_rsi, price_over_200, sma50_over_200 , perc_change = get_tenge_performance()
logger.info(f"Ticker: {ticker}, Decision: {decision}, Histogram Rise/Fall: {last_histogram:.2f}, Last Signal: {last_signal:.2f}, RSI: {last_rsi}, Price/200 {price_over_200}, sma 50 / 200 {sma50_over_200}  , % Change: {perc_change:.2f}")