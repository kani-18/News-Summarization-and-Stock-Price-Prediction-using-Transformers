import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import random

# Mock news data for demonstration purposes
MOCK_NEWS_DB = {
    "AAPL": [
        "Apple announced a record-breaking quarter with iPhone sales surging past expectations in Asian markets.",
        "Supply chain issues continue to plague Apple's new Mac production, potentially delaying shipments by weeks.",
        "Apple's new unreleased VR headset is generating massive hype among developers and enthusiasts.",
        "A major security flaw was found in iOS, prompting an emergency update and user panic."
    ],
    "TSLA": [
        "Tesla deliveries beat estimates, pushing the stock up as production in Gigafactories ramps up efficiently.",
        "Elon Musk announced a delay in the Cybertruck production, citing unforeseen engineering challenges.",
        "A new software update for Tesla's Autopilot has received mixed reviews, with some reporting dangerous bugs.",
        "Tesla expands its supercharger network across Europe, securing a dominant position in the EV market."
    ],
    "MSFT": [
        "Microsoft Azure reports strong growth, taking significant market share from competitors in the cloud space.",
        "The acquisition of a major gaming studio by Microsoft faces regulatory hurdles in multiple countries.",
        "Microsoft's new AI features integrated into Office 365 are receiving widespread praise for boosting productivity."
    ],
    "DEFAULT": [
        "The market is showing mixed signals today as tech stocks rally while energy sectors decline.",
        "Investors are cautious ahead of the upcoming Federal Reserve meeting regarding interest rates.",
        "A new breakthrough in quantum computing was announced, but practical applications seem far off."
    ]
}

def fetch_historical_stock_data(ticker: str, days: int = 30) -> pd.DataFrame:
    """
    Fetches historical stock data using yfinance.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        # Ensure Date timezone isn't problematic
        if df['Date'].dt.tz is not None:
             df['Date'] = df['Date'].dt.tz_localize(None)
             
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_latest_news(ticker: str, k: int = 3) -> list:
    """
    Fetches mock latest news for a given ticker.
    In a real app, this would use an API like NewsAPI, Alpaca, or scrape Yahoo Finance.
    """
    news_pool = MOCK_NEWS_DB.get(ticker.upper(), MOCK_NEWS_DB["DEFAULT"])
    # Randomly select up to k news items
    return random.sample(news_pool, min(k, len(news_pool)))

if __name__ == "__main__":
    # Test data pipeline
    print("Testing data pipeline...")
    df = fetch_historical_stock_data("AAPL", 5)
    print("Stock Data:\n", df.head())
    
    news = fetch_latest_news("AAPL")
    print("\nNews:")
    for n in news:
        print("-", n)
