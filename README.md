> [Agent handoff](HANDOFF.md) — current direction, branch, decisions, verification and next work.

# Grokking Deep Learning — Reproduksi Bab 1–15

Proyek tugas individu kelas Deep Learning berdasarkan Andrew W. Trask, *Grokking Deep Learning* (Manning, 2019). Lima belas notebook berisi rangkuman teori, implementasi yang ditulis mandiri, hasil eksekusi, dan refleksi keterbatasan. Ini **bukan** salinan teks maupun listing kode buku. Akhiran `deep-learning` pada nama repository mengikuti penamaan tugas kelas.

| Bab | Notebook | Ringkasan |
|---|---|---|
| 1. Introducing Deep Learning | [Notebook](Chapter_01_Introducing_Deep_Learning/notebook.ipynb) | Motivasi deep learning, prediksi vs training, peran parameter dan NumPy. Bab pengantar yang terutama konseptual; kode hanya ilustrasi. |
| 2. Fundamental Concepts | [Notebook](Chapter_02_Fundamental_Concepts/notebook.ipynb) | AI ⊃ ML ⊃ DL, supervised/unsupervised, parametric/nonparametric, serta generalisasi. Contoh linear dan kNN sederhana memperjelas teori. |
| 3. Forward Propagation | [Notebook](Chapter_03_Forward_Propagation/notebook.ipynb) | Satu/banyak input dan output, dot product, bentuk matriks, hidden layer, dan kebutuhan aktivasi nonlinier. |
| 4. Gradient Descent | [Notebook](Chapter_04_Gradient_Descent/notebook.ipynb) | Predict–compare–learn, turunan squared error, pemeriksaan gradien numerik, learning rate, konvergensi/divergensi. |
| 5. Generalizing Gradient Descent | [Notebook](Chapter_05_Generalizing_Gradient_Descent/notebook.ipynb) | Gradien vektor dan matriks untuk banyak input/output, efek membekukan bobot, dan training full-batch. |
| 6. Backpropagation | [Notebook](Chapter_06_Backpropagation/notebook.ipynb) | Data lampu jalan pedagogis, batas model linear pada XOR, jaringan dua layer dengan ReLU, sigmoid/BCE, chain rule, dan evaluasi kritis hasil training. |
| 7. Visualizing Neural Networks | [Notebook](Chapter_07_Visualizing_Neural_Networks/notebook.ipynb) | Diagram layer, dimensi matriks, jumlah parameter, dan kesetaraan diagram–persamaan–kode. Bab yang terutama konseptual. |
| 8. Regularization and Batching | [Notebook](Chapter_08_Regularization_and_Batching/notebook.ipynb) | Train/validation/test, overfitting, early stopping, inverted dropout, serta mini-batch pada digit 8×8 offline. |
| 9. Activation Functions | [Notebook](Chapter_09_Activation_Functions/notebook.ipynb) | ReLU/sigmoid/tanh, turunan, softmax stabil, cross-entropy, pemeriksaan gradien, dan klasifikasi tiga kelas sintetis. |
| 10. Convolutional Neural Networks | [Notebook](Chapter_10_Convolutional_Neural_Networks/notebook.ipynb) | Weight sharing, feature map, dimensi/jumlah parameter, backprop kernel, dan CNN kecil pada digit 8×8 offline. |
| 11. Natural Language Processing | [Notebook](Chapter_11_Natural_Language_Processing/notebook.ipynb) | Tokenisasi, vocabulary, embedding sentimen, akumulasi gradien token, cosine similarity, dan keterbatasan bag-of-words. |
| 12. Recurrent Neural Networks | [Notebook](Chapter_12_Recurrent_Neural_Networks/notebook.ipynb) | State recurrent, bobot transisi bersama, BPTT, pemeriksaan gradien, clipping, dan generasi pola karakter. |
| 13. Deep Learning Framework | [Notebook](Chapter_13_Deep_Learning_Framework/notebook.ipynb) | Reverse-mode autograd skalar, graph bercabang, akumulasi gradien, layer, loss, dan optimizer SGD. |
| 14. LSTM | [Notebook](Chapter_14_LSTM/notebook.ipynb) | Memory/gates, character modeling mini, optimasi finite difference, sampling, dan konsep truncated BPTT. |
| 15. Federated Learning | [Notebook](Chapter_15_Federated_Learning/notebook.ipynb) | FedAvg berbobot, spam sintetis, contoh kebocoran gradien, pembatalan mask, dan penjelasan enkripsi homomorfik. |

## Menjalankan

Direkomendasikan Python 3.12. Dari root repository:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/execute_notebooks.py
```

Buka notebook `.ipynb` di VS Code/Jupyter atau Google Colab dan jalankan sel secara berurutan. Bab 8 dan 10 memakai dataset digit 8×8 yang dibundel dalam scikit-learn; tidak ada dataset eksternal yang harus diunduh. Generator sumber notebook ada di `scripts/build_notebooks.py`; jika mengubah generator, jalankan generator lalu `execute_notebooks.py` untuk memperbarui output. Kedua skrip menerima nomor bab opsional, misalnya `python scripts/execute_notebooks.py 7 8`. Random seed membuat contoh dapat diulang, meskipun detail angka/plot dapat sedikit berubah dengan versi paket berbeda.

Untuk membangun ulang Bab 11–15, jalankan `python scripts/build_advanced_notebooks.py`, lalu `python scripts/execute_notebooks.py 11 12 13 14 15`. Skrip generator lanjutan juga menerima nomor bab opsional. Bab 11–15 memakai data kecil buatan sendiri tanpa unduhan. Implementasinya merupakan eksperimen konsep inti: autograd Bab 13 terbatas pada skalar; Bab 14 memakai finite difference untuk melatih LSTM mini, bukan autograd/BPTT LSTM lengkap; Bab 15 mengilustrasikan pembatalan mask dan menjelaskan enkripsi homomorfik tanpa mengimplementasikan kriptografinya. Hasil tidak setara benchmark IMDB/Shakespeare buku.

## Metode, sumber, dan integritas akademik

- **Acuan utama:** Trask, A. W. (2019), *Grokking Deep Learning*, Manning, Bab 1–15. Tautan PDF dibagikan pengajar di sheet tugas; PDF tidak disertakan di repo demi menghormati hak cipta.
- **Cakupan:** penjelasan dan eksperimen konsep inti Bab 1–15, bukan transkripsi lengkap seluruh listing atau halaman buku. Bab 1–2 dan 7 didominasi teori sesuai sifat babnya. Bab 16 belum disertakan.
- **Contoh data:** array kecil/sintetis, serta dataset digit offline dari scikit-learn pada Bab 8 dan 10 sebagai adaptasi dari MNIST buku. Empat pola lampu jalan pada Bab 6 mengikuti masalah pedagogis buku, bukan pengamatan lalu lintas nyata. Bab 6 memakai sigmoid dan binary cross-entropy sebagai adaptasi pedagogis dari contoh output linear/squared error di buku; aturan chain rule tetap dijabarkan.
- **Batas interpretasi:** kecocokan sempurna pada empat contoh training tidak membuktikan generalisasi. Tidak ada klaim keselamatan jalan atau kinerja model nyata. Kode dan uraian ditulis secara mandiri, dengan atribusi ke sumber konsep.
