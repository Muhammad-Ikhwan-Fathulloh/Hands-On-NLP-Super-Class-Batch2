import json
import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Load the vectorizer and vector data
vectorizer = joblib.load("vectorizer.joblib")

with open("vector.json", "r") as file:
    question_vectors = json.load(file)

# Set threshold for distance
THR = 1  # Adjust this value based on testing and requirement

def find_best_match(user_question):
    # Transform the user question into a TF-IDF vector
    user_vector = vectorizer.transform([user_question]).toarray()

    closest_distance = float("inf")
    closest_question = None
    closest_answer = None

    for item in question_vectors:
        question_vector = np.array(item["vector"]).reshape(1, -1)
        distance = euclidean_distances(user_vector, question_vector)[0][0]

        # Check if this question is the closest and below the threshold
        if distance < closest_distance and distance <= THR:
            closest_distance = distance
            closest_question = item["question"]
            closest_answer = item.get("answer", "Maaf, tidak ada jawaban yang tersedia.")

    return closest_question, closest_answer, closest_distance

# Main Q&A loop
print("Selamat datang di sistem tanya jawab Emerald Mabel. Ketik 'exit' untuk keluar.")

while True:
    user_question = input("Anda: ")
    if user_question.lower() == "exit":
        print("Terima kasih telah menggunakan layanan kami. Sampai jumpa!")
        break

    closest_question, closest_answer, closest_distance = find_best_match(user_question)

    if closest_question:
        print(f"Pertanyaan terkait: {closest_question}")
        print(f"Emerald Mabel: {closest_answer}")
    else:
        print("Maaf, kami tidak menemukan jawaban yang sesuai dengan pertanyaan Anda.")