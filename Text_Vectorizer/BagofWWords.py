# Langkah 1: Mengimpor CountVectorizer dari modul feature extraction sklearn
from sklearn.feature_extraction.text import CountVectorizer

# Langkah 2: Membuat daftar dokumen (teks) yang akan dianalisis
dokumen = [
    "saya suka machine learning", 
    "machine learning adalah masa depan",
    "natural language processing adalah masa depan",
    ]

# Langkah 3: Inisialisasi CountVectorizer
# Alat ini akan mengubah teks kita menjadi model "bag of words" dengan menghitung jumlah kemunculan kata
vectorizer = CountVectorizer()

# Langkah 4: Melakukan fitting CountVectorizer pada dokumen dan mengubahnya menjadi vektor
# Langkah ini:
# - Mempelajari semua kata unik dalam dokumen (vocabulary/kosa kata)
# - Membuat matriks di mana setiap baris adalah dokumen dan setiap kolom adalah kata
# - Mengisi matriks ini dengan jumlah kemunculan kata untuk setiap dokumen
vektor = vectorizer.fit_transform(dokumen)

# Langkah 5: Menampilkan matriks jumlah kata yang dihasilkan sebagai array
# Output menunjukkan berapa kali setiap kata muncul di setiap dokumen
print(vektor.toarray())

# Langkah 6 (opsional): Menampilkan kosa kata untuk referensi
# Ini menunjukkan pemetaan setiap kata unik ke indeks kolom dalam matriks
print(vectorizer.get_feature_names_out())
