# IMDB Movie Review Sentiment Analysis

This project implements a sentiment analysis model for IMDB movie reviews. It leverages various natural language processing (NLP) techniques for text preprocessing and machine learning models for classification. The final model (Logistic Regression) is saved along with its TF-IDF vectorizer for easy deployment and inference.

## Project Structure

-   `IMDB Dataset.csv`: The dataset containing movie reviews and their sentiments.
-   `model.pkl`: The trained Logistic Regression model.
-   `vectorizer.pkl`: The fitted TF-IDF vectorizer.
-   `requirements.txt`: Lists all the Python dependencies required to run the project.
-   `app.py`: A Python script that loads the trained model and vectorizer, preprocesses text, and predicts sentiment for new reviews.

## Setup and Installation

To set up the project locally, follow these steps:

1.  **Clone the repository (if applicable) or download the project files.**

2.  **Install Python dependencies:**

    It's recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Ensure NLTK data is downloaded:**

    The `app.py` script includes checks to download `punkt` and `stopwords` data if they are not already present.

## Running the Sentiment Predictor (`app.py`)

The `app.py` script provides a simple interface to test the trained sentiment analysis model.

1.  **Execute the `app.py` script:**

    ```bash
    python app.py
    ```

2.  **Example Usage:**

    The script will print sentiment predictions for a few predefined sample reviews:

    ```
    Review: 'This movie was absolutely fantastic! I loved every minute of it.'
    Predicted Sentiment: positive

    Review: 'Terrible film, a complete waste of time and money. Do not recommend.'
    Predicted Sentiment: negative

    Review: 'The movie had an interesting plot but the acting was mediocre.'
    Predicted Sentiment: negative
    ```

## Models Used

This project explored and evaluated the following classification models:

*   **Multinomial Naive Bayes**
*   **Logistic Regression** (selected as the final model)
*   **Decision Tree**
*   **Linear Support Vector Machine (SVM)** (skipped due to computational intensity for the dataset size)
*   **K-Nearest Neighbors (KNN)** (skipped due to computational intensity for the dataset size)

The Logistic Regression model, trained on TF-IDF features extracted from cleaned text, demonstrated a good balance of accuracy and computational efficiency for this task.
