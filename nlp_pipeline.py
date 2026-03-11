from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class NLPPipeline:
    def __init__(self):
        """
        Initializes the NLP pipelines.
        Downloads the models if not locally cached.
        """
        print("Initializing NLP models. This might take a moment if downloading...")
        
        # Check for GPU
        self.device = 0 if torch.cuda.is_available() else -1
        # Also check for MPS (Apple Silicon) if using PyTorch >= 1.12
        if torch.backends.mps.is_available():
            self.device = "mps"
            
        # Summarizer model: sshleifer/distilbart-cnn-12-6
        model_name = "sshleifer/distilbart-cnn-12-6"
        self.sum_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if self.device != -1:
            self.sum_model = self.sum_model.to(self.device)
        
        # Sentiment Analyzer: ProsusAI/finbert
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert",
            device=self.device
        )
        print("NLP models loaded successfully.")

    def summarize_text(self, text: str) -> str:
        """
        Summarizes a given block of text.
        """
        if not text or len(text.strip()) == 0:
            return ""
            
        # Handle maximum length constraints roughly
        max_length = min(130, max(30, int(len(text.split()) * 0.6)))
        min_length = min(30, max(10, int(len(text.split()) * 0.2)))
        
        try:
            inputs = self.sum_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
            if self.device != -1:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            summary_ids = self.sum_model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            summary = self.sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            return text  # Fallback to original text

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes financial sentiment of the text.
        Returns a dict e.g., {'label': 'positive', 'score': 0.95}
        """
        if not text or len(text.strip()) == 0:
            return {'label': 'neutral', 'score': 0.0}
            
        try:
            result = self.sentiment_analyzer(text)
            return result[0]
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return {'label': 'neutral', 'score': 0.0}

if __name__ == "__main__":
    # Quick test
    nlp = NLPPipeline()
    sample_text = "The company reported a massive increase in revenue for Q3, blowing past analyst expectations. However, future guidance was slightly lowered due to supply chain concerns."
    
    print("\nOriginal:", sample_text)
    summary = nlp.summarize_text(sample_text)
    print("Summary:", summary)
    
    sentiment = nlp.analyze_sentiment(summary)
    print("Sentiment:", sentiment)
