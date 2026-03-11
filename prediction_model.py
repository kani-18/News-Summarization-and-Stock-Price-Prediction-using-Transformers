import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class StockTransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super(StockTransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer predicts a single regression value (price change percentage or next price)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_projection(x) # shape: [batch_size, seq_len, d_model]
        
        # Pass through transformer encoder
        encoded_x = self.transformer_encoder(x) # shape: [batch_size, seq_len, d_model]
        
        # Pool taking the last item in the sequence
        last_item = encoded_x[:, -1, :] # shape: [batch_size, d_model]
        
        # Predict
        out = self.output_layer(last_item) # shape: [batch_size, 1]
        return out


class StockPredictor:
    def __init__(self, seq_length: int = 5):
        self.seq_length = seq_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Features: Open, High, Low, Close, Volume, SentimentScore
        self.input_dim = 6 

    def prepare_data(self, stock_df: pd.DataFrame, sentiment_score: float):
        """
        Takes raw historical df and a current aggregated sentiment score.
        For demo purposes, we will append the same sentiment score to recent days,
        or simulate varying sentiment.
        """
        # Ensure enough data
        if len(stock_df) < self.seq_length:
            return None
            
        data = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Standardize features
        scaled_data = self.scaler.fit_transform(data)
        
        # Add sentiment as a feature (simplified: same sentiment over the sequence for demo)
        sentiment_col = np.full((len(scaled_data), 1), sentiment_score)
        
        # Combine
        combined_features = np.hstack([scaled_data, sentiment_col])
        return combined_features

    def train_mock(self):
        """
        Initializes the model without real training for demonstration.
        In a real scenario, this would load pre-trained weights or train on years of data.
        """
        self.model = StockTransformerModel(input_dim=self.input_dim)
        self.model.eval()
        self.is_trained = True
        print("Mock model initialized with random weights for demo.")

    def predict_next_day(self, stock_df: pd.DataFrame, sentiment_score: float) -> dict:
        """
        Predicts if the next day will be up or down.
        """
        if not self.is_trained:
            self.train_mock()
            
        features = self.prepare_data(stock_df, sentiment_score)
        if features is None:
            return {"status": "error", "message": "Not enough data"}
            
        # Take the most recent seq_length days
        recent_seq = features[-self.seq_length:]
        
        # Convert to tensor: shape [1, seq_length, input_dim]
        input_tensor = torch.FloatTensor(recent_seq).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Simplified interpretation: positive output = bullish, negative = bearish
        prediction_val = output.item()
        
        # Let's adjust based heavily on sentiment to show the pipeline working
        # If sentiment is highly positive (> 0.5), push towards Bullish
        adjusted_val = prediction_val + (sentiment_score * 0.5)
        
        trend = "Bullish ⭐" if adjusted_val > 0 else "Bearish 📉"
        confidence = min(abs(adjusted_val) * 100, 99.9) # Mock confidence score
        
        return {
            "status": "success",
            "trend": trend,
            "raw_value": adjusted_val,
            "confidence": f"{confidence:.1f}%"
        }

if __name__ == "__main__":
    # Test predictor
    from data_pipeline import fetch_historical_stock_data
    df = fetch_historical_stock_data("AAPL", 10)
    
    predictor = StockPredictor()
    # Mock positive sentiment +0.8
    result = predictor.predict_next_day(df, 0.8)
    print("Prediction with Positive sentiment:", result)
