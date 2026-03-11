# News Summarization and Stock Price Prediction using Transformers

This project fetches financial news and historical stock data for a given ticker, summarizes the news using BART, extracts market sentiment using FinBERT, and leverages a Transformer-based model to predict short-term stock price movements. The results are presented in an interactive Streamlit dashboard.

## Features

- **Data Pipeline**: Fetches historical stock data via `yfinance` and retrieves relevant financial news.
- **NLP Pipeline**:
  - **Summarization**: Uses `facebook/bart-large-cnn` to summarize long news articles.
  - **Sentiment Analysis**: Uses `ProsusAI/finbert` to extract market sentiment (positive, negative, neutral) from the summarized text.
- **Prediction Model**: Combines historical stock prices (Open, High, Low, Close, Volume) with aggregated daily sentiment scores, feeding them into a Transformer encoder for time-series forecasting to predict the next day's price movement.
- **Streamlit Dashboard**: A user-friendly interface to input stock tickers and date ranges, view the latest news with summaries and sentiment, explore interactive candlestick charts, and see short-term prediction indicators.

## Prerequisites

- Python 3.8+
- PyTorch

## Installation

1. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
2. Install the necessary dependencies:
    bash
    pip install -r requirements.txt
3. Usage
    To start the Streamlit application, run the following command from the root of the project:

bash
streamlit run app.py
This will launch a local web server (typically at http://localhost:8501) where you can interact with the app. Enter a valid stock ticker (e.g., AAPL, TSLA, MSFT) and select a date range to fetch data, generate summaries, and view predictions.

Project Structure
data_pipeline.py: Handles fetching historical stock prices and news data.
nlp_pipeline.py: Houses the summarization (BART) and sentiment analysis (FinBERT) models.
prediction_model.py: Implements the data preparation and Transformer encoder model for price prediction.
app.py: The main Streamlit dashboard application file.
requirements.txt: Project dependencies.
