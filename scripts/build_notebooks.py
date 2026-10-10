"""Build original learning notebooks inspired by chapters 1-6 of Trask's book.

The code is independently written for this assignment, not a transcription of
the book's listings. Re-run execute_notebooks.py to attach verified outputs.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
M = lambda source: ("markdown", source)
C = lambda source: ("code", source)


def write_chapter(number, slug, cells):
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(source)
        if kind == "markdown"
        else nbf.v4.new_code_cell(source)
        for kind, source in cells
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    path = ROOT / f"Chapter_{number:02d}_{slug}" / "notebook.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    print(path.relative_to(ROOT))


write_chapter(1, "Introducing_Deep_Learning", [
    M("""# Bab 1 — Introducing Deep Learning: Why You Should Learn It

**Acuan:** Andrew W. Trask, *Grokking Deep Learning* (Manning, 2019), Bab 1, hlm. 3–8. Bab ini terutama pengantar konseptual; kode di bawah adalah ilustrasi orisinal, bukan listing dari buku.

**Tujuan belajar:** menjelaskan masalah yang ingin dipecahkan deep learning, mengapa NumPy/Python dipakai, dan batas antara demonstrasi sederhana dengan sistem nyata."""),
    M("""## Intuisi: dari input ke prediksi

Model menerima representasi numerik suatu pengamatan $x$ dan menghasilkan prediksi $\\hat y=f_\\theta(x)$. Simbol $\\theta$ adalah parameter yang bisa disesuaikan dari contoh. Pada neural network, komposisi operasi berbobot dan fungsi nonlinier membentuk representasi bertingkat. *Deep* mengacu pada banyak tahap/lapisan representasi, bukan jaminan bahwa model selalu lebih akurat.

Contoh aplikasi meliputi gambar, teks, dan suara, tetapi kualitas model tetap bergantung pada data, label, tujuan, dan evaluasi. Buku mulai dari aritmetika sederhana lalu membangun ke jaringan berlapis; notebook ini mengikuti urutan tersebut."""),
    C("""import numpy as np

# Ilustrasi paling kecil: satu input, satu bobot, tanpa training.
aktivitas = np.array([1.0, 2.0, 3.0])
bobot_awal = 0.4
prediksi = aktivitas * bobot_awal
print('Input:', aktivitas)
print('Prediksi sebelum belajar:', prediksi)
assert prediksi.shape == aktivitas.shape"""),
    M("""## Apa arti 'belajar'?

Bobot awal di atas hanya pilihan manusia. Belajar berarti mengubah bobot berdasarkan selisih antara prediksi dan target yang diketahui. Secara umum: **prediksi → ukur kesalahan → perbarui parameter → ulangi**. Tanpa target atau tujuan yang jelas, perkalian tadi belum merupakan proses learning. Bab berikutnya membahas jenis pembelajaran; Bab 4–6 menghitung pembaruan bobotnya.

Perlu dibedakan **training** dan **inference**. Training menggunakan data contoh untuk mencari parameter. Inference memakai parameter yang sudah dipelajari untuk memprediksi input baru. Keberhasilan di training tidak otomatis berarti kemampuan generalisasi."""),
    C("""target = np.array([0.5, 1.0, 1.5])  # target sintetis untuk ilustrasi
mse_awal = np.mean((prediksi - target) ** 2)
print('MSE sebelum training:', round(float(mse_awal), 4))
print('Ini baru evaluasi; bobot belum diubah.')"""),
    M("""## Persiapan dan batasan praktis

Python mengatur alur program; NumPy menyediakan array dan operasi matriks yang akan dipakai di bab-bab berikutnya. Bentuk (*shape*) data sangat penting: jumlah fitur input harus cocok dengan dimensi bobot, dan jumlah output harus cocok dengan target. Contoh buku sengaja kecil agar mekanisme dapat dihitung tangan. Di praktik, sebelum melatih model juga diperlukan pemisahan train/test, pemeriksaan data, dan pemilihan metrik yang sesuai.

### Ringkasan

1. Deep learning adalah keluarga metode machine learning berbasis neural network berlapis.
2. Prediksi saja bukan pembelajaran; diperlukan target/tujuan dan aturan pembaruan.
3. Model besar bukan pengganti data yang relevan atau evaluasi pada data baru.

**Refleksi:** kapan model linear sederhana sudah cukup? Kita baru perlu jaringan lebih kompleks jika hubungan input–target memang menuntut kapasitas tambahan, dan manfaatnya terbukti pada evaluasi terpisah."""),
])


write_chapter(2, "Fundamental_Concepts", [
    M("""# Bab 2 — Fundamental Concepts: How Do Machines Learn?

**Acuan:** Trask, *Grokking Deep Learning*, Bab 2, hlm. 9–19. Bab ini terutama teori; contoh numerik berikut adalah adaptasi mandiri untuk memperjelas konsep, bukan salinan listing buku.

**Tujuan belajar:** membedakan AI/ML/DL; supervised vs unsupervised; parametric vs nonparametric; dan memahami bahwa pilihan metode ditentukan oleh struktur masalah."""),
    M("""## 1. Peta konsep

Artificial intelligence (AI) adalah payung luas untuk sistem yang melakukan tugas yang kita anggap memerlukan kecerdasan. Machine learning (ML) mempelajari aturan dari data alih-alih hanya aturan eksplisit. Deep learning (DL) adalah bagian ML yang memakai neural network berlapis. Hubungan himpunannya: $DL\\subset ML\\subset AI$. Tidak semua AI menggunakan ML, dan tidak semua ML menggunakan DL.

Pada **supervised learning**, tersedia pasangan $(x_i,y_i)$; model mempelajari peta $x\\mapsto y$. Pada **unsupervised learning**, target $y$ tidak diberikan dan kita mencari struktur, misalnya kelompok atau representasi. Kategori ini mendeskripsikan informasi yang tersedia saat training, bukan jenis file atau bentuk output."""),
    C("""import numpy as np

# Data sintetis: temperatur (x) dan konsumsi energi (y) berlabel.
x = np.array([18., 20., 22., 24., 26.])
y = np.array([31., 30., 29., 31., 34.])
print('Supervised: jumlah pasangan (x,y) =', len(x))
print('Unsupervised: hanya x tersedia, bentuk =', x.shape)
assert x.shape == y.shape"""),
    M("""## 2. Parametric vs nonparametric

Model **parametric** menetapkan bentuk/fungsi dengan jumlah parameter terhingga sebelum melihat ukuran dataset. Misalnya $\\hat y=wx+b$ selalu punya dua parameter untuk satu fitur; data bertambah, jumlah parameter tetap. Model **nonparametric** lebih fleksibel: kompleksitas efektif dapat tumbuh bersama data, misalnya k-nearest neighbors perlu menyimpan contoh training. Istilah nonparametric tidak berarti 'tanpa parameter'—$k$ dan aturan jarak tetap harus dipilih.

Kedua pendekatan punya trade-off: asumsi kuat dapat memberi efisiensi saat data sedikit tetapi menimbulkan bias jika bentuk salah; pendekatan lebih fleksibel dapat menangkap pola rumit namun rentan noise dan biaya penyimpanan/prediksi."""),
    C("""# Parametric: kuadrat terkecil untuk y ≈ w*x + b.
design = np.column_stack([x, np.ones_like(x)])
w, b = np.linalg.lstsq(design, y, rcond=None)[0]

# Nonparametric: prediksi dari dua tetangga terdekat (k=2).
def knn_predict(query, train_x, train_y, k=2):
    nearest = np.argsort(np.abs(train_x - query))[:k]
    return float(np.mean(train_y[nearest]))

query = 23.0
print('Linear (2 parameter):', round(float(w * query + b), 2))
print('kNN (menyimpan 5 contoh):', round(knn_predict(query, x, y), 2))"""),
    M("""## 3. Dari konsep menuju neural network

Jaringan di bab berikutnya adalah model parametric: arsitektur menentukan jumlah bobot. Jika satu layer mengubah $d$ fitur ke $h$ unit, matriks bobot berukuran $d\\times h$ (ditambah bias jika dipakai). Sebuah neuron menghitung kombinasi berbobot, lalu mungkin menerapkan aktivasi. Meskipun jumlah parameter tetap untuk arsitektur tertentu, neural network bisa berkapasitas sangat besar.

**Generalisasi** adalah kemampuan memberi prediksi baik pada data baru dari proses yang sama. Model yang meniru setiap contoh training tetapi gagal pada contoh baru mengalami overfitting. Oleh karena itu, evaluasi perlu data yang tidak dipakai untuk menyesuaikan parameter atau hyperparameter."""),
    C("""# Tunjukkan hubungan dimensi pada model parametric berlapis.
n_features, n_hidden, n_outputs = 3, 4, 1
parameter_count = n_features * n_hidden + n_hidden * n_outputs
print('Bobot dua layer, tanpa bias:', parameter_count)
assert parameter_count == 16"""),
    M("""### Ringkasan dan refleksi

- Supervised/unsupervised membedakan ada-tidaknya target eksplisit saat training.
- Parametric/nonparametric membedakan apakah ukuran representasi model ditentukan sebelumnya atau dapat bertumbuh bersama data.
- Neural network adalah salah satu model parametric, bukan sinonim dari semua ML.
- Contoh di atas sengaja kecil dan sintetis; tidak cukup untuk membandingkan akurasi metode secara ilmiah."""),
])
