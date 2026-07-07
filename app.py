import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Ensure NLTK data is downloaded for deployment environment
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Get stopwords once
stop_words = set(stopwords.words('english'))

# Define the preprocessing function
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text)
    # Lowercase and remove punctuation (keeping only alphabetic words)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(text):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    
    # Vectorize the cleaned text
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)
    
    # Decode prediction
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    
    return sentiment

if __name__ == '__main__':
    # Example usage
    sample_review_positive = "This movie was absolutely fantastic! I loved every minute of it."
    sample_review_negative = "Terrible film, a complete waste of time and money. Do not recommend."
    sample_review_neutral = "The movie had an interesting plot but the acting was mediocre."
    
    print(f"Review: '{sample_review_positive}'\nPredicted Sentiment: {predict_sentiment(sample_review_positive)}\n")
    print(f"Review: '{sample_review_negative}'\nPredicted Sentiment: {predict_sentiment(sample_review_negative)}\n")
    print(f"Review: '{sample_review_neutral}'\nPredicted Sentiment: {predict_sentiment(sample_review_neutral)}\n")
