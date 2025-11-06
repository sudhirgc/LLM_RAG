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
import mplfinance as mpf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def confirm_stock_ticker(ticker:str)->str:
    """ 
    This method confirms that the ticker exists and if the ticker is not found, 
    we assume it be a company and get ticker for that company
    Args:
        Stock ticker or comapany name.
    """
    logger.info(f'Input received -> {ticker}')
    try:
        ticker = ticker.strip()

        df = pd.read_csv('us_symbols.csv')
        #print(df.info())
        #print(df.head())
    
        #Lets see of we have the Tracker in the list 
        if df['ticker'].eq(ticker.upper()).any():
            logger.info('We have exact match in the CSV, Returning the same as output')
            return ticker.upper()
        # We are here implies that either ticker does not exist or is a company name... 
        if df['name'].str.contains(ticker,case=False,na=False).any():
            # We need to get the matching row and return the ticker column value
            df = df[df['name'].str.contains(ticker,case=False,na=False)]
            if (len(df) > 0):
                # We have match as company name, returning the ticker
                ticker = df['ticker'].iloc[0]
                logger.info(f'We matched from company name, returning tracker as {ticker}, company name {df['name'].iloc[0]}')
                return ticker
            else:
                #This could be a new ticker or company name is misspelled.. 
                logger.info(f'We could NOT match any company or ticker name -> {ticker}')
        else:
            logger.info(f'We could NOT match any company or ticker name -> {ticker}')
    except Exception as e:
        logger.error('confirm_stock_ticker::Error getting Ticker -> ', e)
        raise e
    return ticker

def get_stock_performance(ticker:str):
    """
    This method gets the stock performance for the specified stock ticker as percentage
    Args:
        ticker: The stock ticker for which performance needs to be determined
    """
    logger.info(f'Input received in get_stock_performance, Ticker -> {ticker}')

    # Lets  confirm the ticker, convert to ticker if needed......
    try:
        ticker = confirm_stock_ticker(ticker=ticker)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        #df = yfinance.download(tickers=str(ticker), start=start_date, end=end_date)

        #df = yfinance.download(tickers=str(ticker),period='ytd')
        ticker_obj = yfinance.Ticker(ticker=ticker)
        df = ticker_obj.history(start=start_date, end=end_date)
        print(df.head())

        print(df.info())
        df.dropna(inplace=True)

        mpf.plot(df, type='candle', mav=(8,50,200) , style='charles', volume=True)

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

        adx_indicator = ADXIndicator(high=df['High'].squeeze(), 
                                     low=df['Low'].squeeze(), 
                                     close=df['Close'].squeeze())
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()

        rsi_indicator = RSIIndicator(close=df['Close'].squeeze())
        df['rsi'] = rsi_indicator.rsi()

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(df.index.values, df['Close'].values, "y--", label=f"{ticker}")
        ax.plot(df.index.values, df['SMA_200'].values, "c--", label="SMA_200")
        ax.plot(df.index.values, df['adx'].values, "k--", label="ADX")
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
        last_adx = round(df['adx'].iloc[-1])
        last_rsi = round(df['rsi'].iloc[-1])

        last_stock_price = df['Close'].iloc[-1]
        last_200_price = df['SMA_200'].iloc[-1]
        last_50_price = df['SMA_50'].iloc[-1]

        decision = 'Hold'
        is_clear_trend = False
        if last_adx >= 25:
            is_clear_trend = True

        if last_histogram < 0 and is_clear_trend:
            #This is tending towards Sell, just confirm that Sell is supported by adx
            if last_rsi > 70:
                decision = 'Sell'
            else:
                decision = 'Possible Sell'
        elif last_histogram > 0 and is_clear_trend:
            if last_rsi < 30:
                decision = 'Buy'
            else:
                decision = 'Possible Buy'
        elif last_histogram < 0:
            if last_rsi > 70:
                decision = 'Possible Sell'
        elif last_histogram > 0:
            if last_rsi < 30:
                decision = 'Possible Buy'

        

        

        start_val = df['Close'].iloc[0] 
        end_val = df['Close'].iloc[-1] 
        print( start_val, end_val)
        perc_change = (end_val - start_val)*100/start_val 
        print(f" Change in Percentage - {perc_change}")

        return ticker, decision, last_histogram, last_macd, last_signal, last_rsi, last_adx, perc_change
    except Exception as e:
        logger.error('get_stock_performance :: Error calculating performance -> ', e)
    return ticker, 'Hold', 0, 0, 0, 0, 0, 0


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    # This is the input from the USER -> 
    ticker, decision, histogram, macd, signal, rsi, adx, perc_change = get_stock_performance(message.content)

    # Send a response back to the user
    await cl.Message(
        content=f"Ticker: {ticker}, Decision: {decision}, MACD Rise/Fall: {histogram:.2f}, RSI: {rsi}, ADX: {adx}, % Change: {perc_change:.2f}",
    ).send()


logger.info(get_stock_performance('USDKZT=X'))