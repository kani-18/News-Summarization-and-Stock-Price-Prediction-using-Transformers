import streamlit as st
import pandas as pd
import plotly.graph_objects as GO
import time
import os
import yfinance as yf

# Suppress some transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set page config
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e2129;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #00c0f2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .news-card {
        background-color: #1e2129;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-top: 2px solid #333;
    }
    .sentiment-positive { color: #00e676; font-weight: bold; }
    .sentiment-negative { color: #ff1744; font-weight: bold; }
    .sentiment-neutral { color: #bdbdbd; font-weight: bold; }
    .stSpinner > div > div { border-color: #00c0f2 !important; border-bottom-color: transparent !important; }
</style>
""", unsafe_allow_html=True)

# Application Initialization
@st.cache_resource
def load_nlp():
    from nlp_pipeline import NLPPipeline
    return NLPPipeline()

@st.cache_resource
def load_predictor():
    from prediction_model import StockPredictor
    return StockPredictor()

from data_pipeline import fetch_historical_stock_data, fetch_latest_news

# Load Models
nlp = load_nlp()
predictor = load_predictor()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135673.png", width=60)
    st.title("AI Stock Predictor")
    st.markdown("Combines **News Summarization**, **Sentiment Analysis**, and **Transformer Models** to predict short term movement.")
    
    st.markdown("### Settings")
    AVAILABLE_TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    ticker = st.selectbox("Stock Ticker", options=AVAILABLE_TICKERS, index=0)
    days = st.slider("Historical Data Days", min_value=10, max_value=90, value=30, step=10)
    num_news = st.slider("News Articles to Analyze", min_value=1, max_value=5, value=3)
    
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

if run_btn:
    # 1. Fetching Data
    st.header(f"Analysis for {ticker}")
    
    with st.status("Fetching live data...", expanded=True) as status:
        st.write("Fetching historical market data...")
        stock_df = fetch_historical_stock_data(ticker, days=days)
        time.sleep(0.5) # UX delay
        st.write("Fetching latest news...")
        news_items = fetch_latest_news(ticker, k=num_news)
        status.update(label="Data fetching complete!", state="complete", expanded=False)

    if stock_df.empty:
        st.error(f"Failed to fetch data for ticker: {ticker}. Try AAPL, TSLA, MSFT.")
    else:
        # Layout columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Candlestick chart
            st.subheader("Price History")
            fig = GO.Figure(data=[GO.Candlestick(x=stock_df['Date'],
                            open=stock_df['Open'],
                            high=stock_df['High'],
                            low=stock_df['Low'],
                            close=stock_df['Close'])])
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=10, b=0),
                height=400,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Transformer Prediction (Next Day)")
            
            # 2. NLP Analysis
            st.write("Analyzing news sentiment...")
            aggregated_score = 0
            processed_news = []
            
            my_bar = st.progress(0)
            for i, text in enumerate(news_items):
                # Summarize
                summary = nlp.summarize_text(text)
                # Analyze sentiment
                sentiment = nlp.analyze_sentiment(summary)
                
                # Convert sentiment string to numeric score
                score = sentiment['score']
                if sentiment['label'].lower() == 'negative':
                    score = -score
                elif sentiment['label'].lower() == 'neutral':
                    score = 0
                    
                aggregated_score += score
                
                processed_news.append({
                    "original": text,
                    "summary": summary,
                    "sentiment": sentiment['label'],
                    "score": score
                })
                my_bar.progress((i + 1) / len(news_items))
            
            my_bar.empty()
            
            # Finalize average sentiment
            avg_sentiment = aggregated_score / len(news_items) if news_items else 0
            
            # 3. Model Prediction
            prediction = predictor.predict_next_day(stock_df, avg_sentiment)
            
            # Display Prediction Widget
            if prediction["status"] == "success":
                sentiment_color = "#00e676" if avg_sentiment > 0 else "#ff1744" if avg_sentiment < 0 else "#bdbdbd"
                
                # UI Card
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin-top:0;">Overall Sentiment</h3>
                    <h1 style="color:{sentiment_color};">{avg_sentiment:.2f}</h1>
                    <p>Calculated from recent news using FinBERT.</p>
                    <hr>
                    <h3 style="margin-top:0;">Model Forecast</h3>
                    <h1 style="color:#00c0f2;">{prediction['trend']}</h1>
                    <p>Confidence: {prediction['confidence']}</p>
                    <p style="font-size:12px;color:#888;">Input sequence: {predictor.seq_length} days + Sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(prediction["message"])

        # Display News Analysis Section
        st.subheader("News Analysis Detail")
        for idx, item in enumerate(processed_news):
            color_class = f"sentiment-{item['sentiment'].lower()}"
            st.markdown(f"""
            <div class="news-card">
                <h4>📰 Article {idx+1}</h4>
                <p><b>Original:</b> {item['original']}</p>
                <p><b>BART Summary:</b> <i>{item['summary']}</i></p>
                <p><b>FinBERT Sentiment:</b> <span class="{color_class}">{item['sentiment'].upper()}</span> ({item['score']:.2f})</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Empty State
    st.info("👈 Enter a stock ticker and click **Run Analysis** to start.")
    
    # Showcase architecture on start page
    st.markdown("### Architecture Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. News Summarization")
        st.write("Uses `BART` (facebook/bart-large-cnn) to condense long financial documents.")
    with col2:
        st.markdown("#### 2. Sentiment Extractor")
        st.write("Feeds summaries into `FinBERT` to map financial context to market sentiment polarity.")
    with col3:
        st.markdown("#### 3. Time-Series Transformer")
        st.write("A PyTorch `TransformerEncoder` merges OHLCV data with sentiment for robust next-day forecasting.")

