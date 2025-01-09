# Langkah 1: Mengimpor TfidfVectorizer dari modul feature extraction sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Langkah 2: Membuat daftar dokumen (teks) yang akan dianalisis
dokumen = ["saya suka machine learning", "machine learning adalah masa depan"]

# Langkah 3: Inisialisasi TfidfVectorizer
# Alat ini akan mengubah teks kita menjadi representasi "TF-IDF" untuk setiap kata
vectorizer = TfidfVectorizer()

# Langkah 4: Melakukan fitting TfidfVectorizer pada dokumen dan mengubahnya menjadi vektor
# Langkah ini:
# - Mempelajari semua kata unik dalam dokumen (vocabulary/kosa kata)
# - Membuat matriks di mana setiap baris adalah dokumen dan setiap kolom adalah kata
# - Mengisi matriks ini dengan skor TF-IDF untuk setiap kata dalam setiap dokumen
vektor = vectorizer.fit_transform(dokumen)

# Langkah 5: Menampilkan matriks TF-IDF yang dihasilkan sebagai array
# Output menunjukkan skor TF-IDF untuk setiap kata dalam setiap dokumen
print(vektor.toarray())

# Langkah 6 (opsional): Menampilkan kosa kata untuk referensi
# Ini menunjukkan pemetaan setiap kata unik ke indeks kolom dalam matriks
print(vectorizer.get_feature_names_out())