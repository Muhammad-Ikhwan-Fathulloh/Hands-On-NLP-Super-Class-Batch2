import pandas as pd
from pandarallel import pandarallel

# Inisialisasi pandarallel dengan progress bar
pandarallel.initialize(progress_bar=True)

# Fungsi untuk menghitung panjang karakter dari teks
def count_text_length(text):
    return len(text)

# Menjalankan kode utama di dalam blok ini
if __name__ == "__main__":
    # Membuat contoh DataFrame
    data = pd.DataFrame({
        'text': [
            'Contoh teks satu.',
            'Teks kedua yang lebih panjang.',
            'Ini adalah teks ketiga.',
            'Sebuah teks lain untuk pengujian.',
            'Pandarallel dapat mempercepat proses ini.'
        ]
    })

    # Menggunakan pandarallel untuk menghitung panjang teks secara paralel
    data['text_length'] = data['text'].parallel_apply(count_text_length)

    # Menampilkan DataFrame hasil
    print(data)
