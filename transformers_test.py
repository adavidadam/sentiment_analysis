from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
print(sentiment_pipeline("This is a test"))
