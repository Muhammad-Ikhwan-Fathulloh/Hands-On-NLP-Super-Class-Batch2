# Langkah 1: Mengimpor pustaka Word2Vec dari Gensim
from gensim.models import Word2Vec

# Langkah 2: Membuat daftar kalimat yang akan dianalisis
# Setiap kalimat adalah daftar kata yang merepresentasikan sebuah dokumen atau frasa
kalimat = [
        ["saya", "suka", "machine", "learning"], 
        ["machine", "learning", "adalah", "masa", "depan"],
        ["natural", "language", "processing", "adalah", "masa", "depan"]
    ]

# Langkah 3: Membuat model Word2Vec
# Parameter yang digunakan:
# - vector_size=100: Ukuran vektor embedding untuk setiap kata adalah 100 dimensi
# - window=5: Jumlah kata yang diperhitungkan di sekitar kata target dalam satu kalimat
# - min_count=1: Kata yang muncul minimal satu kali akan diproses
# - workers=4: Jumlah proses yang digunakan untuk mempercepat training
model = Word2Vec(sentences=kalimat, vector_size=100, window=5, min_count=1, workers=4)

# Langkah 4: Menampilkan vektor embedding untuk kata tertentu
# "model.wv" adalah tempat di mana vektor kata disimpan setelah training.
# Contoh ini menampilkan vektor dari kata "machine"
print("Vector dari Kata machine")
print(model.wv["machine"])

print("Vector dari Kata natural")
print(model.wv["natural"])
