# Grokking Deep Learning — Reproduksi Bab 1–6

Proyek tugas individu kelas Deep Learning berdasarkan Andrew W. Trask, *Grokking Deep Learning* (Manning, 2019). Enam notebook berisi rangkuman teori, implementasi NumPy yang ditulis mandiri, hasil eksekusi, dan refleksi keterbatasan. Ini **bukan** salinan teks maupun listing kode buku. Akhiran `deep-learning` pada nama repository mengikuti penamaan tugas kelas.

| Bab | Notebook | Ringkasan |
|---|---|---|
| 1. Introducing Deep Learning | [Notebook](Chapter_01_Introducing_Deep_Learning/notebook.ipynb) | Motivasi deep learning, prediksi vs training, peran parameter dan NumPy. Bab pengantar yang terutama konseptual; kode hanya ilustrasi. |
| 2. Fundamental Concepts | [Notebook](Chapter_02_Fundamental_Concepts/notebook.ipynb) | AI ⊃ ML ⊃ DL, supervised/unsupervised, parametric/nonparametric, serta generalisasi. Contoh linear dan kNN sederhana memperjelas teori. |
| 3. Forward Propagation | [Notebook](Chapter_03_Forward_Propagation/notebook.ipynb) | Satu/banyak input dan output, dot product, bentuk matriks, hidden layer, dan kebutuhan aktivasi nonlinier. |
| 4. Gradient Descent | [Notebook](Chapter_04_Gradient_Descent/notebook.ipynb) | Predict–compare–learn, turunan squared error, pemeriksaan gradien numerik, learning rate, konvergensi/divergensi. |
| 5. Generalizing Gradient Descent | [Notebook](Chapter_05_Generalizing_Gradient_Descent/notebook.ipynb) | Gradien vektor dan matriks untuk banyak input/output, efek membekukan bobot, dan training full-batch. |
| 6. Backpropagation | [Notebook](Chapter_06_Backpropagation/notebook.ipynb) | Data lampu jalan pedagogis, batas model linear pada XOR, jaringan dua layer dengan ReLU, sigmoid/BCE, chain rule, dan evaluasi kritis hasil training. |

## Menjalankan

Direkomendasikan Python 3.12. Dari root repository:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/execute_notebooks.py
```

Buka notebook `.ipynb` di VS Code/Jupyter atau Google Colab dan jalankan sel secara berurutan. Semua contoh hanya memerlukan NumPy dan Matplotlib; tidak ada dataset eksternal yang harus diunduh. Generator sumber notebook ada di `scripts/build_notebooks.py`; jika mengubah generator, jalankan generator lalu `execute_notebooks.py` untuk memperbarui output. Random seed membuat contoh dapat diulang, meskipun detail angka/plot dapat sedikit berubah dengan versi paket berbeda.

## Metode, sumber, dan integritas akademik

- **Acuan utama:** Trask, A. W. (2019), *Grokking Deep Learning*, Manning, Bab 1–6. Tautan PDF dibagikan pengajar di sheet tugas; PDF tidak disertakan di repo demi menghormati hak cipta.
- **Cakupan:** penjelasan dan implementasi konsep inti Bab 1–6, bukan transkripsi lengkap seluruh listing atau halaman buku. Bab 1–2 didominasi teori sesuai sifat babnya. Bab 7–10 sengaja belum disertakan karena tenggatnya terpisah.
- **Contoh data:** array kecil/sintetis. Empat pola lampu jalan pada Bab 6 mengikuti masalah pedagogis buku, bukan pengamatan lalu lintas nyata. Bab 6 memakai sigmoid dan binary cross-entropy sebagai adaptasi pedagogis dari contoh output linear/squared error di buku; aturan chain rule tetap dijabarkan.
- **Batas interpretasi:** kecocokan sempurna pada empat contoh training tidak membuktikan generalisasi. Tidak ada klaim keselamatan jalan atau kinerja model nyata. Kode dan uraian ditulis secara mandiri, dengan atribusi ke sumber konsep.
