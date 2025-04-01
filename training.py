import pandas as pd
import string
import numpy as np

# Load the CSV data
# Read the CSV, skipping lines with too many fields
# Make sure the path 'Sentiment Analysis Dataset.csv' is correct
try:
    dataRaw = pd.read_csv("Sentiment Analysis Dataset.csv", on_bad_lines='skip')
except FileNotFoundError:
    print("Error: 'Sentiment Analysis Dataset.csv' not found.")
    print("Please make sure the file is in the same directory as your script or provide the full path.")
    # Exit or handle the error appropriately
    exit()


# Assuming 'Sentiment' column contains numerical labels (e.g., 0 for negative, 1 for positive)
# and 'SentimentText' contains the text data.
ListSentimentSocre = dataRaw["Sentiment"].tolist()
ListSentence = dataRaw["SentimentText"].tolist()

# --- Your Preprocessing Code ---
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize text (split by spaces) - Note: Vectorizers often handle tokenization
    # words = text.split()
    # For standard vectorizers, returning the processed string is often better
    return text # Return the processed string

# Apply preprocess_text to each sentence in the ListSentence
# We'll apply this *within* the vectorizer later, or just use the vectorizer's built-in preprocessing
# For simplicity now, let's work with the original ListSentence and let the vectorizer handle basic preprocessing.
# If you want to use your specific tokenization, vectorizers can sometimes accept pre-tokenized input
# or a custom preprocessor/tokenizer function.

# ListSentenceProcessed = [preprocess_text(sentence) for sentence in ListSentence] # You can use this if needed

# Convert all score into float for training (and potentially to int for classification)
# Often classification models in sklearn expect integer labels
try:
    ListSentimentSocre = np.array(ListSentimentSocre).astype(int)
except ValueError as e:
    print(f"Error converting scores to integers: {e}")
    print("Please check the 'Sentiment' column for non-numeric values.")
    # Consider inspecting unique values: print(dataRaw['Sentiment'].unique())
    exit()

print(f"Loaded {len(ListSentence)} sentences and {len(ListSentimentSocre)} scores.")
print("Sample processed sentence (using vectorizer's default):", ListSentence[0])
print("Sample score:", ListSentimentSocre[0])
print("-" * 20)

# --- Step-by-Step Model Training ---

# Import necessary libraries from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving the model and vectorizer

# Step 1: Split Data into Training and Testing Sets
# -------------------------------------------------
# Purpose: We need to evaluate our model on data it hasn't seen during training.
# We split the original sentences (ListSentence) and the corresponding scores (ListSentimentSocre).
# test_size=0.2 means 20% of the data will be used for testing, 80% for training.
# random_state ensures reproducibility - the split will be the same each time you run the code.

print("Step 1: Splitting data...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    ListSentence,
    ListSentimentSocre,
    test_size=0.2,
    random_state=42 # Use a fixed number for reproducibility
)

print(f"Training set size: {len(X_train_text)} sentences")
print(f"Test set size: {len(X_test_text)} sentences")
print("-" * 20)

# Step 2: Feature Extraction (Vectorization)
# ------------------------------------------
# Purpose: Machine learning models need numerical input, not raw text.
# We'll use TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors.
# TF-IDF represents words based on their frequency in a document adjusted by how common they are across all documents.
# - We initialize the TfidfVectorizer. It handles lowercasing and tokenization by default.
#   You could pass your preprocess_text function using the `preprocessor` argument if needed.
# - We 'fit' the vectorizer *only* on the training data (X_train_text). This learns the vocabulary.
# - We then 'transform' both the training and test data into TF-IDF vectors.
# Important: Use fit_transform on training data, but only transform on test data to avoid data leakage.

print("Step 2: Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000, # Limit the vocabulary size to the top 5000 words (optional, helps performance)
    stop_words='english' # Remove common English stop words (like 'the', 'a', 'is')
    # You could add: preprocessor=preprocess_text if you want your specific cleaning
)

# Fit the vectorizer on the training text and transform it
X_train_tfidf = vectorizer.fit_transform(X_train_text)

# Transform the test text using the *same* fitted vectorizer
X_test_tfidf = vectorizer.transform(X_test_text)

print(f"Shape of TF-IDF matrix for training data: {X_train_tfidf.shape}")
print(f"Shape of TF-IDF matrix for test data: {X_test_tfidf.shape}")
# The shape shows (number_of_documents, number_of_features/words_in_vocabulary)
print("-" * 20)

# Step 3: Choose and Train a Model
# --------------------------------
# Purpose: Select a machine learning algorithm and train it on the vectorized training data and corresponding labels.
# We'll use Logistic Regression, a common and effective baseline for text classification.
# Other options include Naive Bayes (MultinomialNB), Support Vector Machines (SVC), etc.
# The `fit` method trains the model.

print("Step 3: Training a Logistic Regression model...")
model = LogisticRegression(max_iter=1000) # Increase max_iter if it fails to converge

# Train the model using the TF-IDF vectors and the training labels
model.fit(X_train_tfidf, y_train)

print("Model training complete.")
print("-" * 20)

# Step 4: Evaluate the Model
# --------------------------
# Purpose: Assess how well the trained model performs on the unseen test data.
# We use the trained model to predict labels for the test set's TF-IDF vectors.
# Then we compare these predictions (y_pred) with the actual labels (y_test).

print("Step 4: Evaluating the model...")
# Predict sentiments on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print a detailed classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Rows: Actual class, Columns: Predicted class
# [[True Negative, False Positive],
#  [False Negative, True Positive]] (for binary classification 0/1)
print("-" * 20)

# Step 5: Make Predictions on New Data (Example)
# ---------------------------------------------
# Purpose: Use the trained model and the fitted vectorizer to predict the sentiment of new, unseen sentences.
# Remember to apply the same preprocessing/vectorization steps.

print("Step 5: Making predictions on new sentences...")
new_sentences = [
    "This movie was absolutely fantastic, I loved it!",
    "I was really disappointed with the product.",
    "It's an okay experience, not great but not terrible either.",
    "What a waste of time and money."
]

# 1. Preprocess (if your vectorizer doesn't do it implicitly or if you used a custom one)
#    Our TfidfVectorizer handles lowercasing, so we just pass the raw sentences.
#    If you used preprocess_text earlier, apply it here:
#    new_sentences_processed = [preprocess_text(s) for s in new_sentences]

# 2. Vectorize using the *same* fitted vectorizer
new_sentences_tfidf = vectorizer.transform(new_sentences)

# 3. Predict using the trained model
new_predictions = model.predict(new_sentences_tfidf)
new_predictions_proba = model.predict_proba(new_sentences_tfidf) # Get probabilities

for sentence, prediction, proba in zip(new_sentences, new_predictions, new_predictions_proba):
    sentiment = "Positive" if prediction == 1 else "Negative" # Adjust based on your label mapping (0/1)
    confidence = proba[prediction] # Probability of the predicted class
    print(f"Sentence: '{sentence}'")
    print(f"Predicted Sentiment: {sentiment} (Label: {prediction}), Confidence: {confidence:.4f}")
    print("---")

print("-" * 20)

# Step 6: Save the Model and Vectorizer (Optional but Recommended)
# ---------------------------------------------------------------
# Purpose: Save your trained model and the vectorizer so you can reload and use them later without retraining.

print("Step 6: Saving the model and vectorizer...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model saved as 'sentiment_model.pkl'")
print("Vectorizer saved as 'tfidf_vectorizer.pkl'")
print("-" * 20)

# --- How to Load and Use the Saved Model Later ---
# print("\nLoading and using the saved model...")
# loaded_model = joblib.load('sentiment_model.pkl')
# loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
# print("Model and vectorizer loaded.")

# example_sentence = ["This is a test sentence to classify."]
# example_tfidf = loaded_vectorizer.transform(example_sentence)
# prediction = loaded_model.predict(example_tfidf)
# print(f"Prediction for '{example_sentence[0]}': {'Positive' if prediction[0] == 1 else 'Negative'}")