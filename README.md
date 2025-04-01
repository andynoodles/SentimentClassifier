# Basic Sentiment Analysis Project

This project demonstrates building, training, and evaluating a simple sentiment analysis model using Python, pandas, and scikit-learn. It takes text data with corresponding sentiment labels, preprocesses the text, trains a Logistic Regression classifier using TF-IDF features, evaluates its performance, and provides functionality to predict sentiment on new text samples.

**Project Status:** (As of Tuesday, April 1, 2025) - Development/Example Implementation

## Features

* Loads data from a CSV file (`Sentiment Analysis Dataset.csv`).
* Handles potential errors during CSV reading (skips bad lines, checks for file existence).
* Splits the dataset into training and testing sets (80/20 split).
* Preprocesses text data (default: lowercasing, removing punctuation via TF-IDF's built-in options).
* Vectorizes text data using TF-IDF (`TfidfVectorizer`) with stop word removal.
* Trains a Logistic Regression model on the vectorized training data.
* Evaluates the model's performance on the test set using:
    * Accuracy score
    * Classification Report (Precision, Recall, F1-Score)
    * Confusion Matrix
* Demonstrates how to predict the sentiment of new, unseen text sentences.
* Saves the trained Logistic Regression model (`sentiment_model.pkl`) and the fitted TF-IDF vectorizer (`tfidf_vectorizer.pkl`) using joblib for later use.

## Dataset

This project requires the `Sentiment Analysis Dataset.csv` file to be present in the root directory.

Dataset available at:
```
https://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip 
```

**Important:** The dataset is **not included** in this repository. You need to provide your own dataset.

The expected format is a CSV file with at least two columns:
1.  A column containing the text data (e.g., named "SentimentText").
2.  A column containing the numerical sentiment labels (e.g., named "Sentiment"). The code currently assumes binary classification where labels are convertible to integers (e.g., 0 for negative, 1 for positive).

## Requirements

The required Python libraries are listed in `requirements.txt`.

* `pandas`
* `numpy`
* `scikit-learn`
* `joblib`

## Installation

1.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Linux/macOS
    python -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Add the dataset:** Place your `Sentiment Analysis Dataset.csv` file in the project's root directory.

## Usage

To train your model run

```bash
python training.py
```

To evaluate using my model 

```bash
python eval.py
```

Created and tested by 111590037 資工三 陳奕翔 

Assisted by GPT-4o mini, Gemini2.5 pro