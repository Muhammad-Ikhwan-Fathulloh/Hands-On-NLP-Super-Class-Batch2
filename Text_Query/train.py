import json
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load Indonesian Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to preprocess text: lowercase, remove punctuation, and lemmatize
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lemmatize each word
    words = text.split()
    lemmatized_words = [stemmer.stem(word) for word in words]
    # Join the words back into a single string
    return ' '.join(lemmatized_words)

# Load the corpus
with open("data.json", "r") as file:
    corpus = json.load(file)["qa_corpus"]

# Extract questions and answers
questions = [item["question"] for item in corpus]
answers = [item["answer"] for item in corpus]

# Preprocess questions and answers
preprocessed_questions = [preprocess_text(question) for question in questions]
preprocessed_answers = [preprocess_text(answer) for answer in answers]

# Combine preprocessed questions and answers into a single corpus for TF-IDF training
combined_corpus = preprocessed_questions + preprocessed_answers

# Initialize and fit the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(combined_corpus)

# Save the vectorizer model
joblib.dump(vectorizer, "vectorizer.joblib")

# Transform each question into a TF-IDF vector
question_vectors = vectorizer.transform(preprocessed_questions)

# Prepare data for saving to vector.json
vector_data = []
for question, answer, vector in zip(questions, answers, question_vectors):
    vector_data.append({
        "question": question,
        "answer": answer,
        "vector": vector.toarray().tolist()[0]  # Convert sparse matrix to dense list
    })

# Save the vectors to vector.json
with open("vector.json", "w") as file:
    json.dump(vector_data, file, indent=4)

print("Training complete and data saved to vectorizer.joblib and vector.json")