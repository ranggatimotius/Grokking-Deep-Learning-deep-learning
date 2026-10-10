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


write_chapter(3, "Forward_Propagation", [
    M("""# Bab 3 — Introduction to Neural Prediction: Forward Propagation

**Acuan:** Trask, *Grokking Deep Learning*, Bab 3, hlm. 21–46. Implementasi di sini ditulis mandiri dengan contoh angka kecil yang sejalan dengan urutan konsep buku. Semua output adalah prediksi **sebelum training**.

**Tujuan belajar:** menghitung prediksi dari satu input, banyak input, banyak output, hingga dua layer; memeriksa shape matriks; dan memahami peran bobot serta aktivasi."""),
    M("""## 1. Satu input → satu output

Neuron paling sederhana tanpa bias maupun aktivasi menghitung $\\hat y=xw$. Bobot $w$ menyatakan sensitivitas prediksi terhadap input. Mengalikan dengan bobot bukan 'berpikir'; tanpa proses belajar, nilainya hanya pemetaan yang kita tentukan. Nilai output juga belum tentu probabilitas walaupun persoalan aslinya kelas biner."""),
    C("""import numpy as np

x = 8.5        # fitur sintetis: rata-rata 'toes' tim
w = 0.12       # bobot awal, belum dilatih
y_hat = x * w
print(f'Prediksi = {x} × {w} = {y_hat:.3f}')
assert np.isclose(y_hat, 1.02)"""),
    M("""## 2. Banyak input → satu output

Untuk $d$ fitur, $\\hat y=\\sum_{j=1}^{d} x_jw_j=x^\\top w$. Ini disebut *weighted sum* atau dot product. Setiap fitur harus dipasangkan dengan bobot yang tepat; urutan dan skala fitur berpengaruh. Jika dua fitur sangat berbeda skala, besar bobot mentah tidak bisa langsung dibandingkan sebagai 'kepentingan'."""),
    C("""features = np.array([8.5, 0.65, 1.2])  # toes, win/loss record, fans (toy)
weights = np.array([0.10, 0.20, -0.10])
manual = sum(float(feature * weight) for feature, weight in zip(features, weights))
vectorized = float(features @ weights)
print('Hitung manual:', round(manual, 3), 'NumPy dot:', round(vectorized, 3))
assert np.isclose(manual, vectorized) and features.shape == weights.shape"""),
    M("""## 3. Satu input → banyak output

Jika satu angka input perlu menghasilkan $k$ prediksi, gunakan vektor bobot $(w_1,\\ldots,w_k)$ dan $\\hat{\\boldsymbol y}=x\\boldsymbol w$. Tanpa hidden layer, masing-masing output terpisah: mengubah bobot output pertama tidak langsung mengubah output lain. Output dapat mewakili besaran berbeda, sehingga masing-masing memerlukan target dan skala/loss yang tepat saat training."""),
    C("""record = 0.65
output_weights = np.array([0.30, 0.20, 0.90])
out = record * output_weights
for label, prediction in zip(['win', 'sad', 'hurt'], out):
    print(label, round(float(prediction), 3))
assert out.shape == (3,)"""),
    M("""## 4. Banyak input → banyak output

Satukan bobot dalam matriks $W\\in\\mathbb{R}^{d\\times k}$. Untuk satu contoh $x\\in\\mathbb{R}^{d}$, hasil $xW\\in\\mathbb{R}^{k}$. Untuk $n$ contoh sekaligus, $X\\in\\mathbb{R}^{n\\times d}$ memberi $XW\\in\\mathbb{R}^{n\\times k}$. Baris matriks input = contoh; kolom = fitur. Pemeriksaan dimensi adalah pertahanan pertama terhadap bug perhitungan yang tampak masuk akal namun salah."""),
    C("""W = np.array([[.10, .05], [.20, -.10], [-.10, .30]])  # 3 fitur × 2 output
X = np.array([[8.5, .65, 1.2], [9.5, .80, 1.3]])
Y_hat = X @ W
print('X:', X.shape, 'W:', W.shape, 'prediksi:', Y_hat.shape)
print(Y_hat.round(3))
assert Y_hat.shape == (2, 2)
assert np.allclose(Y_hat[0], features @ W)"""),
    M("""## 5. Prediksi di atas prediksi: hidden layer

Dengan dua matriks, $H=\\phi(XW_1+b_1)$ dan $\\hat Y=HW_2+b_2$. $H$ adalah representasi intermediate, bukan target yang diamati. Tanpa aktivasi nonlinier, $(XW_1)W_2=X(W_1W_2)$ sehingga dua layer linear setara satu layer linear. ReLU $\\phi(z)=\\max(0,z)$ memperkenalkan nonlinieritas, tetapi juga dapat mematikan unit jika selalu menerima nilai negatif. Di bab ini bobot masih belum dioptimalkan."""),
    C("""rng = np.random.default_rng(3)
W1 = rng.normal(0, .2, size=(3, 4))
W2 = rng.normal(0, .2, size=(4, 2))
hidden = np.maximum(0, X @ W1)
deep_prediction = hidden @ W2
linear_equivalent = X @ (W1 @ W2)
print('hidden:', hidden.shape, 'output:', deep_prediction.shape)
print('Jarak output nonlinear vs linear:', round(float(np.linalg.norm(deep_prediction - linear_equivalent)), 5))
assert hidden.shape == (2, 4) and deep_prediction.shape == (2, 2)"""),
    M("""### Ringkasan dan refleksi

- Forward propagation mengalirkan input melalui operasi berbobot menuju prediksi, tanpa mengubah bobot.
- Dot product dan perkalian matriks adalah bentuk ringkas dari weighted sums.
- Dimensi $n\\times d$, $d\\times h$, dan $h\\times k$ harus konsisten.
- Banyak layer linear saja tidak menambah jenis fungsi yang bisa direpresentasikan; aktivasi nonlinier penting untuk pola kompleks.
- Output mentah di contoh ini bukan probabilitas terkalibrasi dan belum mengukur akurasi pada data baru."""),
])


write_chapter(4, "Gradient_Descent", [
    M("""# Bab 4 — Introduction to Neural Learning: Gradient Descent

**Acuan:** Trask, *Grokking Deep Learning*, Bab 4, hlm. 47–77. Contoh perhitungan dan grafik ditulis ulang secara mandiri.

**Tujuan belajar:** menghubungkan error dengan perubahan bobot, memahami turunan, learning rate, konvergensi/divergensi, dan membedakan hafalan satu contoh dari generalisasi."""),
    C("""import numpy as np
import matplotlib.pyplot as plt

x, target = 2.0, 1.0
weight = 0.1
def squared_error(weight, x=x, target=target):
    return (x * weight - target) ** 2

print('Prediksi awal:', x * weight)
print('Squared error awal:', squared_error(weight))"""),
    M("""## 1. Predict → compare → learn

Untuk satu contoh, gunakan $L(w)=(xw-y)^2$. Selisih $\\delta=\\hat y-y$ menyimpan **arah** kesalahan. Turunan eksak $dL/dw=2(xw-y)x=2\\delta x$. Banyak demonstrasi buku menyerap faktor 2 ke learning rate, sehingga bisa memakai $\\delta x$; keduanya benar jika learning rate disesuaikan. Di notebook ini kita memakai turunan eksak dan menuliskan konvensinya secara eksplisit."""),
    C("""prediction = x * weight
delta = prediction - target
analytic_gradient = 2 * delta * x
epsilon = 1e-6
numeric_gradient = (squared_error(weight + epsilon) - squared_error(weight - epsilon)) / (2 * epsilon)
print('Delta:', round(delta, 4))
print('Gradien analitik:', round(analytic_gradient, 4))
print('Gradien beda hingga:', round(numeric_gradient, 4))
assert np.isclose(analytic_gradient, numeric_gradient, atol=1e-6)"""),
    M("""## 2. Dari hot-and-cold ke gradien

Metode *hot-and-cold* membandingkan loss setelah bobot dinaikkan dan diturunkan sedikit, lalu memilih arah yang lebih baik. Ini memperkirakan arah, tetapi membutuhkan evaluasi tambahan untuk tiap kemungkinan langkah. Turunan memberikan arah dan sensitivitas lokal secara langsung. Pembaruan gradient descent adalah $w_{t+1}=w_t-\\alpha\\,dL/dw$, dengan $\\alpha>0$ sebagai learning rate."""),
    C("""step = 0.01
print('Loss jika bobot diturunkan:', round(squared_error(weight - step), 4))
print('Loss jika bobot dinaikkan:', round(squared_error(weight + step), 4))
alpha = 0.05
updated = weight - alpha * analytic_gradient
print('Bobot baru:', round(updated, 4), 'loss baru:', round(squared_error(updated), 4))
assert squared_error(updated) < squared_error(weight)"""),
    M("""## 3. Beberapa iterasi dan learning rate

Learning rate terlalu kecil bergerak lambat; terlalu besar dapat melompati minimum dan menyebabkan osilasi/divergensi. Untuk fungsi kuadrat sederhana ini, kita dapat melihat efeknya tanpa variabel lain. Konvergensi pada **training example yang sama** tidak membuktikan bahwa bobot bekerja pada input baru."""),
    C("""def train_scalar(alpha, iterations=12):
    w = 0.1
    history = []
    for _ in range(iterations):
        history.append(squared_error(w))
        w -= alpha * 2 * (x * w - target) * x
    return w, np.array(history)

fig, ax = plt.subplots(figsize=(6, 3.5))
for lr in [.01, .10, .40]:
    final_w, history = train_scalar(lr)
    ax.plot(history, marker='o', label=f'alpha={lr}, w={final_w:.2f}')
ax.set(xlabel='Iterasi', ylabel='Squared error', title='Pengaruh learning rate')
ax.set_yscale('symlog', linthresh=.01)
ax.legend(); plt.tight_layout(); plt.show()
print('Bobot optimum untuk satu contoh:', target / x)"""),
    M("""## 4. Dari satu contoh ke beberapa contoh

Jika ada $n$ pasangan, loss rata-rata $L(w)=\\frac1n\\sum_i(wx_i-y_i)^2$ mempunyai gradien $\\frac{2}{n}\\sum_i (wx_i-y_i)x_i$. Gradien ini merangkum tekanan semua contoh; contoh yang menginginkan pembaruan berlawanan arah dapat saling menyeimbangkan. Pembaruan pada seluruh dataset disebut full-batch gradient descent. Kita sengaja tidak menambahkan bias agar formula tetap satu parameter."""),
    C("""inputs = np.array([1., 2., 3., 4.])
goals = np.array([.5, 1., 1.5, 2.])  # hubungan sintetis y=0.5x
w = -0.2
losses = []
for _ in range(150):
    errors = w * inputs - goals
    losses.append(np.mean(errors**2))
    gradient = 2 * np.mean(errors * inputs)
    w -= .02 * gradient
print('Bobot belajar:', round(w, 4), 'MSE akhir:', round(losses[-1], 8))
assert np.isclose(w, .5, atol=1e-3)"""),
    M("""### Ringkasan dan refleksi

- Error memberi besar penyimpangan; gradien memberi arah pembaruan yang menurunkannya secara lokal.
- Turunan bisa diperiksa dengan *finite difference* untuk menangkap kesalahan aljabar/kode.
- Learning rate adalah bagian penting dari algoritme; nilai besar tidak selalu lebih cepat.
- Menghafal satu pasangan input–target berbeda dari mempelajari pola yang berlaku pada contoh baru. Evaluasi generalisasi masih diperlukan."""),
])
