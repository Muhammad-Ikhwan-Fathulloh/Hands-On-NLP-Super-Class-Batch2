# Named Entity Recognition (NER)

**Named Entity Recognition (NER)** adalah salah satu teknik dalam bidang **Natural Language Processing (NLP)** yang digunakan untuk mengidentifikasi dan mengklasifikasikan entitas yang ada dalam sebuah teks. Entitas yang dimaksud biasanya mencakup nama orang, lokasi, organisasi, tanggal, waktu, dan entitas lainnya yang memiliki makna khusus. NER memungkinkan komputer untuk memahami dan mengkategorikan informasi penting yang terkandung dalam teks, yang berguna dalam berbagai aplikasi, seperti pencarian informasi, ekstraksi data, dan sistem tanya jawab otomatis.

## Tujuan dari NER

Tujuan utama dari NER adalah untuk mengekstrak entitas tertentu dari teks sehingga komputer dapat memahami struktur dan informasi yang terkandung di dalamnya. Misalnya, pada kalimat "Bill Gates adalah pendiri Microsoft," NER bertugas mengidentifikasi dan mengkategorikan dua entitas utama: "Bill Gates" (sebagai orang) dan "Microsoft" (sebagai organisasi).

## Kategori Entitas yang Umum Dikenali dalam NER

1. **Person (PER)**: Nama orang, baik individu maupun grup.  
   Contoh: "Bill Gates," "Albert Einstein."
   
2. **Organization (ORG)**: Nama perusahaan, organisasi, atau lembaga.  
   Contoh: "Microsoft," "United Nations."
   
3. **Location (LOC)**: Nama tempat geografis, seperti negara, kota, atau bangunan.  
   Contoh: "Paris," "Indonesia."
   
4. **Date (DATE)**: Informasi tentang waktu, seperti tanggal, hari, atau waktu tertentu.  
   Contoh: "15 Januari 2023," "kemarin."
   
5. **Time (TIME)**: Menyebutkan waktu, seperti jam atau periode waktu.  
   Contoh: "pukul 10 pagi," "sore hari."
   
6. **Miscellaneous (MISC)**: Kategori lain yang tidak termasuk dalam kategori di atas, misalnya nama produk atau peristiwa.  
   Contoh: "Windows 10," "Olimpiade."

## Bagaimana NER Bekerja?

Proses NER biasanya dilakukan dalam beberapa langkah berikut:

1. **Preprocessing Teks**: Teks yang diberikan diproses terlebih dahulu, misalnya dengan tokenisasi (memisahkan teks menjadi kata atau kalimat) dan penghapusan tanda baca.

2. **Penentuan Entitas**: Algoritma NER menggunakan model yang telah dilatih untuk mengidentifikasi entitas dalam teks. Proses ini melibatkan penggunaan teknik seperti:
   - Rule-based systems: Menggunakan aturan dan pola untuk mendeteksi entitas.
   - Machine learning-based models: Menggunakan model pembelajaran mesin, seperti Hidden Markov Models (HMM), Conditional Random Fields (CRF), dan Deep Learning (misalnya model berbasis transformer seperti BERT).

3. **Klasifikasi Entitas**: Setelah entitas diidentifikasi, entitas tersebut diklasifikasikan ke dalam kategori yang sesuai, seperti orang, organisasi, atau lokasi.

4. **Postprocessing**: Menyaring hasil deteksi dan memperbaiki kesalahan yang mungkin terjadi, seperti menggabungkan entitas yang terpisah oleh pemisah, atau memverifikasi kategori entitas.

## Teknik yang Digunakan dalam NER

- **Rule-based NER**: Menggunakan aturan dan pola yang telah ditentukan sebelumnya. Misalnya, untuk mendeteksi tanggal, pola regex bisa digunakan untuk mencocokkan format tanggal tertentu (misalnya, dd/mm/yyyy).
  
- **Machine Learning**: Pendekatan ini melibatkan pelatihan model dengan dataset yang berlabel untuk mengenali entitas. Teknik yang sering digunakan meliputi:
  - Naive Bayes, SVM (Support Vector Machines), atau Random Forests.
  - Conditional Random Fields (CRF) untuk memodelkan ketergantungan antar kata.

- **Deep Learning**: Pendekatan berbasis jaringan saraf dalam, terutama **Recurrent Neural Networks (RNNs)** dan **Transformers** (seperti BERT), yang dapat belajar lebih baik dalam menangani konteks dan ketergantungan kata-kata dalam kalimat. Pendekatan ini lebih canggih karena mampu menangani teks dengan lebih baik dan memahami konteks yang lebih dalam.

## Aplikasi dari NER

1. **Pencarian Informasi**: Menemukan dan mengekstrak informasi penting (misalnya, nama orang atau tempat) dari teks besar atau dokumen.
   
2. **Sistem Tanya Jawab (QA)**: NER digunakan untuk mengenali dan memahami entitas dalam pertanyaan dan jawaban, seperti dalam sistem asisten virtual.

3. **Analisis Sentimen**: Mengidentifikasi entitas dalam ulasan produk atau media sosial untuk mengukur sentimen terkait dengan entitas tertentu.

4. **Pemetaan dan Geolokasi**: Menggunakan entitas tempat untuk membantu menentukan lokasi geografis dalam aplikasi berbasis peta.

5. **Ekstraksi Data**: Mengambil informasi penting secara otomatis dari artikel berita, laporan, atau database.

## Contoh Penggunaan NER

Misalnya, dengan kalimat berikut:  
*"Elon Musk, CEO Tesla, mengunjungi Tokyo pada 10 November 2024 untuk peluncuran produk baru."*

Proses NER akan menghasilkan entitas berikut:

- **Elon Musk** (Person)
- **CEO Tesla** (Organization)
- **Tokyo** (Location)
- **10 November 2024** (Date)

## Tantangan dalam NER

- **Ambiguitas**: Beberapa entitas mungkin memiliki arti ganda, seperti "Apple" yang bisa merujuk pada perusahaan atau buah.
- **Entitas yang Tidak Terlihat**: Beberapa entitas mungkin jarang ditemukan dalam data pelatihan, sehingga model kesulitan mendeteksinya.
- **Konfigurasi Kalimat**: Kalimat yang kompleks atau panjang dapat menyulitkan NER untuk memahami konteks yang tepat.
- **Bahasa dan Budaya**: Entitas dalam bahasa atau konteks budaya tertentu bisa sulit dikenali oleh model yang dilatih di bahasa atau konteks yang berbeda.

## Kesimpulan

NER merupakan teknik yang sangat penting dalam NLP yang membantu mesin untuk memahami teks lebih dalam dan mengidentifikasi informasi yang relevan secara otomatis. Ini berperan penting dalam berbagai aplikasi seperti analisis data besar, pencarian informasi, dan pengolahan dokumen otomatis. Model-model NER terus berkembang seiring dengan kemajuan dalam teknologi pembelajaran mesin dan deep learning.