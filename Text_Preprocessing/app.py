import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Inisialisasi
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('indonesian'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Teks contoh
text = "Saya belajar Natural Language Processing di rumah."

# Tokenisasi
tokens = word_tokenize(text.lower())  # Mengubah teks ke huruf kecil
print("Token:", tokens)

# Menghapus Stopwords
tokens = [word for word in tokens if word not in stop_words]
print("Tanpa Stopwords:", tokens)

# Stemming
stemmed = [stemmer.stem(word) for word in tokens]
print("Hasil Stemming:", stemmed)

# Lemmatization
lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
print("Hasil Lemmatization:", lemmatized)