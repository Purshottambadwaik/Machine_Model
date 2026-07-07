import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Ensure NLTK data is downloaded for deployment environment
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab') # Added punkt_tab download
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

# Load the saved model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Get stopwords once
stop_words = set(stopwords.words('english'))

# Define the preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(text):
    if not text.strip(): # Handle empty input
        return "Please enter some text to predict sentiment."
    cleaned_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return sentiment

# Streamlit UI
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review below to get its sentiment prediction (positive or negative).')

user_input = st.text_area('Movie Review', '')

if st.button('Predict Sentiment'):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.success(f'Predicted Sentiment: **{sentiment.upper()}**')
    else:
        st.warning('Please enter a review to get a prediction.')
