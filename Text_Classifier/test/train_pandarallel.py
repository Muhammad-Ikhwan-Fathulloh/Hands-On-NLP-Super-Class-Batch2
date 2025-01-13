import pandas as pd
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pandarallel import pandarallel

# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

# Initialize Indonesian stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Preprocess function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    print(f"Original text: {text}")  # Debugging print
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    print(f"Processed text: {text}")  # Debugging print
    # Stemming
    text = stemmer.stem(text)
    return text

if __name__ == '__main__':
    # Load data
    data = pd.read_csv("data3.csv")
    
    # Apply parallel preprocessing to the text column
    data['text'] = data['text'].parallel_apply(preprocess_text)

    # Separate features and labels
    X = data['text']
    y = data['intent']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define TF-IDF and Logistic Regression pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the model pipeline with joblib
    joblib.dump(pipeline, 'intent_classification.joblib')
