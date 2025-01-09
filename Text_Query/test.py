# Langkah 1: Mengimpor StemmerFactory dari pustaka Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Langkah 2: Membuat objek stemmer
# Dengan memanfaatkan `StemmerFactory`, kita dapat menghasilkan stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Langkah 3: Contoh proses stemming
# Kita memiliki kalimat dalam bahasa Indonesia yang akan diproses
kalimat = "Mereka menari dengan sangat gembira"

# Menggunakan objek stemmer untuk melakukan proses stemming
hasil = stemmer.stem(kalimat)

# Menampilkan hasil stemming
print(hasil)  # Output: "mereka tari dengan sangat gembira"
