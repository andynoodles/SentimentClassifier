import joblib

print("\nLoading and using the saved model...")
loaded_model = joblib.load('sentiment_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
print("Model and vectorizer loaded.")

example_sentence = ["This is a test sentence to classify."]
example_tfidf = loaded_vectorizer.transform(example_sentence)
prediction = loaded_model.predict(example_tfidf)
print(f"Prediction for '{example_sentence[0]}': {'Positive' if prediction[0] == 1 else 'Negative'}")