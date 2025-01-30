### Named Entity Recognition (NER) dengan IDs to Labels

Dalam konteks **Named Entity Recognition (NER)**, **ids_to_labels** adalah sebuah kamus yang digunakan untuk memetakan ID numerik ke label entitas yang relevan. Dalam hal ini, ID numerik yang digunakan oleh model NER diubah menjadi kategori entitas yang lebih mudah dipahami. Label-label ini mengidentifikasi jenis entitas yang ditemukan dalam teks.

Berikut adalah penjelasan tentang label yang terdapat pada kamus `ids_to_labels` yang diberikan:

| **ID** | **Label** | **Deskripsi** |
|--------|-----------|---------------|
| 0      | B-art     | **B**eginning of **art** (Seni). Mengindikasikan bahwa entitas adalah karya seni atau seni visual (misalnya, nama buku atau lukisan). |
| 1      | B-eve     | **B**eginning of **eve**nt. Menandakan bahwa entitas adalah sebuah peristiwa atau kejadian (misalnya, nama konser atau konferensi). |
| 2      | B-geo     | **B**eginning of **geo**graphic location (lokasi geografis). Ini adalah lokasi geografis seperti negara atau benua. |
| 3      | B-gpe     | **B**eginning of **gpe** (Geopolitical entity). Menandakan entitas yang merujuk pada wilayah politik atau negara (misalnya, negara atau kota). |
| 4      | B-nat     | **B**eginning of **nat**ural entity. Ini adalah entitas yang merujuk pada entitas alamiah, seperti jenis spesies atau fenomena alam. |
| 5      | B-org     | **B**eginning of **org**anization. Ini adalah label untuk organisasi, baik itu perusahaan, lembaga pemerintah, atau organisasi non-profit. |
| 6      | B-per     | **B**eginning of **per**son. Menandakan entitas yang merujuk pada nama orang (misalnya, individu atau tokoh terkenal). |
| 7      | B-tim     | **B**eginning of **tim**e. Menandakan entitas yang berkaitan dengan waktu, seperti tanggal atau periode waktu. |
| 8      | I-art     | **I**nside of **art**. Label ini mengindikasikan bahwa entitas adalah bagian dari karya seni yang disebutkan sebelumnya. |
| 9      | I-eve     | **I**nside of **eve**nt. Menunjukkan bagian dari sebuah peristiwa atau kejadian yang telah dikenali. |
| 10     | I-geo     | **I**nside of **geo**. Menandakan bagian dari lokasi geografis yang telah dikenali (misalnya, bagian dari nama negara atau kota). |
| 11     | I-gpe     | **I**nside of **gpe**. Menunjukkan bagian dari entitas politik atau negara yang telah dikenali. |
| 12     | I-nat     | **I**nside of **nat**ural. Menunjukkan bahwa entitas adalah bagian dari entitas alami yang sudah dikenali. |
| 13     | I-org     | **I**nside of **org**anization. Menandakan bagian dari organisasi yang telah dikenali. |
| 14     | I-per     | **I**nside of **per**son. Menunjukkan bahwa entitas adalah bagian dari individu yang sudah dikenali sebelumnya. |
| 15     | I-tim     | **I**nside of **tim**e. Ini adalah label yang digunakan untuk bagian dari entitas waktu. |
| 16     | O         | **O**utside. Menandakan bahwa token tersebut bukan bagian dari entitas yang dikenali, atau token tersebut adalah kata umum yang tidak terkait dengan entitas tertentu. |

### Penjelasan Label Entitas:

1. **B-art, I-art**: 
   - **B-art** menandakan bahwa token tersebut adalah bagian pertama dari sebuah karya seni, misalnya nama buku atau lukisan.
   - **I-art** adalah token yang mengikuti dan menjadi bagian dari entitas "art", seperti nama album musik atau lukisan.

2. **B-eve, I-eve**: 
   - **B-eve** digunakan untuk memulai entitas yang merujuk pada suatu acara atau peristiwa.
   - **I-eve** digunakan untuk token yang mengikuti entitas yang merujuk pada suatu peristiwa.

3. **B-geo, I-geo**: 
   - **B-geo** merujuk pada lokasi geografis yang baru dimulai, seperti kota, negara, atau benua.
   - **I-geo** menandakan token yang merupakan bagian dari lokasi geografis yang telah dikenali.

4. **B-gpe, I-gpe**: 
   - **B-gpe** digunakan untuk menandai entitas politik atau administratif, seperti negara atau kota.
   - **I-gpe** adalah bagian dari entitas politik yang dikenali.

5. **B-nat, I-nat**: 
   - **B-nat** adalah entitas yang merujuk pada sesuatu yang alami, seperti spesies hewan atau fenomena alam.
   - **I-nat** menandakan bahwa token tersebut adalah bagian dari entitas alami yang lebih besar.

6. **B-org, I-org**: 
   - **B-org** digunakan untuk organisasi, seperti perusahaan atau lembaga.
   - **I-org** menunjukkan token yang merupakan bagian dari organisasi yang telah dikenali.

7. **B-per, I-per**: 
   - **B-per** digunakan untuk menandai nama orang atau individu yang sedang diperkenalkan.
   - **I-per** adalah bagian dari nama orang yang telah dikenali.

8. **B-tim, I-tim**: 
   - **B-tim** adalah token yang menunjukkan entitas waktu yang baru dimulai, seperti tanggal atau periode.
   - **I-tim** menunjukkan token yang merupakan bagian dari entitas waktu yang telah dikenali.

9. **O**: 
   - **O** menandakan token yang tidak termasuk dalam entitas manapun, atau kata yang tidak relevan untuk NER.

### Penjelasan Kode NER

Pada kode di atas, **Named Entity Recognition (NER)** dilakukan dengan menggunakan model pra-latih berbasis *transformer* yang dilengkapi dengan tokenizer untuk memproses teks input.

1. **Tokenisasi Input**: Teks yang diterima (`text`) diproses dengan tokenizer, yang memecah teks menjadi token dan menyelaraskan ID untuk setiap kata atau sub-kata. Tokenizer ini juga menangani padding dan pemangkasan (truncation) untuk memastikan panjang token input sesuai dengan batas maksimum yang ditentukan.

2. **Pemindahan ke Device**: Jika model dapat menggunakan GPU (jika tersedia), maka model dan input data akan dipindahkan ke perangkat yang sesuai (baik GPU atau CPU).

3. **Prediksi**: Model kemudian melakukan prediksi terhadap token yang telah diproses untuk mengklasifikasikan setiap token ke dalam label NER yang sesuai. Label ini merujuk pada kategori entitas yang terdeteksi (misalnya, orang, organisasi, lokasi, dll.).

4. **Mapping IDs ke Label**: Hasil prediksi model (dalam bentuk ID numerik) kemudian dipetakan ke label menggunakan kamus `ids_to_labels`, sehingga label dapat dibaca dan dipahami sebagai entitas yang lebih relevan.

5. **Menyaring dan Mengumpulkan Entitas**: Token yang tidak termasuk dalam entitas atau yang tidak relevan (seperti padding atau token subword) akan disaring. Hanya entitas yang valid (misalnya, nama orang atau organisasi) yang akan disertakan dalam hasil akhir.

### Contoh Output:

Misalnya, jika input teks adalah:

> "Bill Gates bekerja di Microsoft."

Dan jika model memprediksi ID untuk token "Bill" menjadi `6` (B-per), "Gates" menjadi `14` (I-per), dan "Microsoft" menjadi `5` (B-org), maka entitas yang terdeteksi adalah:

- **Bill Gates**: Nama orang (B-per, I-per)
- **Microsoft**: Nama organisasi (B-org)

Hasilnya akan menunjukkan entitas yang terdeteksi seperti:

- `['B-per', 'I-per', 'B-org']`