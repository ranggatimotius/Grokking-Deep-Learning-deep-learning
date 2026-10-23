"""Original, offline chapter 11-15 learning experiments; not book transcriptions."""
from pathlib import Path
import sys
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
selected = set(map(int, sys.argv[1:])) or set(range(11, 16))
def write(number, slug, cells):
    if number not in selected:
        return
    book = nb.v4.new_notebook()
    for i, (kind, source) in enumerate(cells):
        cell = nb.v4.new_code_cell(source) if kind == 'c' else nb.v4.new_markdown_cell(source)
        cell.id = f'chapter-{number}-cell-{i:02d}'
        book.cells.append(cell)
    book.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    path = ROOT / f'Chapter_{number:02d}_{slug}' / 'notebook.ipynb'
    path.parent.mkdir(exist_ok=True)
    nb.write(book, path)
    print(path.relative_to(ROOT))
M = lambda s: ('m', s)
C = lambda s: ('c', s)

write(11, 'Natural_Language_Processing', [
M('''# Bab 11 — Neural Networks That Understand Language
**Acuan:** Andrew W. Trask, *Grokking Deep Learning*, Bab 11 (hlm. 187–208).

Tujuan: mengubah token menjadi indeks dan embedding, melatih klasifikasi sentimen, serta membaca kedekatan vektor. Buku memakai ulasan IMDB dan word analogy; di sini korpus kecil berbahasa Inggris ditulis sendiri untuk eksperimen offline. Hasil tidak mewakili akurasi IMDB.'''),
M('''## Representasi dan supervised NLP
Tokenizer menetapkan unit bahasa yang dilihat model. Vocabulary dipelajari hanya dari training; token baru masuk `<unk>`. Embedding adalah matriks E berukuran vocabulary × dimensi; mengambil baris E setara perkalian one-hot dengan E. Rata-rata embedding memberi representasi kalimat dengan panjang bervariasi, tetapi kehilangan urutan kata: “not good” dan “good not” memiliki rata-rata sama. Label sentimen menentukan arah representasi yang dipelajari, bukan makna universal kata.'''),
C('''import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(11)
texts = ['good enjoyable film', 'great enjoyable story', 'good great acting', 'wonderful good story',
         'bad boring film', 'terrible boring story', 'bad terrible acting', 'awful bad story']
y = np.array([1,1,1,1,0,0,0,0], dtype=float)
vocab = ['<unk>'] + sorted(set(' '.join(texts).split()))
index = {word:i for i,word in enumerate(vocab)}
encode = lambda text: np.array([index.get(w,0) for w in text.lower().split()])
sentences = [encode(t) for t in texts]
E = rng.normal(0,.15,(len(vocab),6)); w = rng.normal(0,.15,6); b = 0.
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z,-30,30)))
print('Vocabulary:', vocab, 'embedding shape:', E.shape)'''),
M('''## Embedding menerima gradien dari target
Untuk kalimat s, h adalah rata-rata embedding dan p=sigmoid(h·w+b). Binary cross-entropy memberi delta p−y. Gradien setiap kemunculan token adalah delta × w / panjang kalimat. Token berulang harus menjumlahkan gradien: `np.add.at` menangani indeks yang sama. Semua gradien dihitung sebelum bobot diubah.'''),
C('''losses=[]
for epoch in range(250):
    gE=np.zeros_like(E); gw=np.zeros_like(w); gb=0.; loss=0.
    for ids,target in zip(sentences,y):
        h=E[ids].mean(axis=0); p=sigmoid(h@w+b); d=p-target
        loss -= target*np.log(p+1e-12)+(1-target)*np.log(1-p+1e-12)
        gw += d*h/len(y); gb += d/len(y)
        np.add.at(gE,ids,d*w/(len(ids)*len(y)))
    E-=.5*gE; w-=.5*gw; b-=.5*gb; losses.append(loss/len(y))
pred=np.array([sigmoid(E[ids].mean(axis=0)@w+b) for ids in sentences])
print('Training accuracy:', np.mean((pred>=.5)==y))
for text in ['good story', 'boring film', 'unknown film']:
    print(text, round(float(sigmoid(E[encode(text)].mean(axis=0)@w+b)),3))
plt.plot(losses); plt.xlabel('Epoch'); plt.ylabel('BCE'); plt.show()
assert losses[-1] < losses[0]'''),
M('''## Kedekatan dan analogi
Cosine similarity membandingkan arah vektor setelah normalisasi. Analogi seperti king−man+woman adalah hipotesis geometris tentang embedding yang dipelajari dari banyak konteks; delapan kalimat ini tidak cukup untuk mengharapkan analogi tersebut. Embedding sentimen bisa mendekatkan kata karena label serupa, bukan karena sinonim. `<unk>` tidak mendapat latihan di korpus ini, sehingga prediksi kalimat dengan token baru perlu ditafsirkan hati-hati.'''),
C('''norm=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-12)
query=index['good']; scores=norm@norm[query]
nearest=np.argsort(scores)[::-1][1:5]
print('Tetangga cosine good:', [(vocab[i],round(float(scores[i]),3)) for i in nearest])
assert np.allclose(E[encode('good bad')].mean(0), E[encode('bad good')].mean(0))'''),
M('''## Ringkasan dan keterbatasan
Embedding dilatih bersama tujuan prediksi; bag-of-words menangani panjang bervariasi namun mengabaikan urutan. Eksperimen menunjukkan mekanisme pembelajaran, dengan akurasi training sebagai pemeriksaan kode. Evaluasi generalisasi memerlukan korpus lebih besar, split terpisah, penanganan negasi, dan pengujian token baru.''')])

write(12, 'Recurrent_Neural_Networks', [
M('''# Bab 12 — Recurrent Layers for Variable-Length Data
**Acuan:** Trask, *Grokking Deep Learning*, Bab 12 (hlm. 209–229).

Tujuan: memahami mengapa urutan diperlukan, membangun recurrent state, dan menghitung backpropagation through time (BPTT). Korpus pola karakter dibuat sendiri, bukan teks Shakespeare buku.'''),
M('''## State dan bobot yang dipakai ulang
RNN menghitung h_t=tanh(x_t W_x+h_(t−1) W_h+b), lalu logits=h_t W_y+c. Matriks transisi W_h digunakan di setiap timestep: jumlah parameter tidak bertambah dengan panjang urutan. Berbeda dari rata-rata embedding, hasil recurrent state bergantung pada urutan. State awal direset antarcontoh independen agar informasi tidak bocor.'''),
C('''import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(12)
alphabet='abc'; tokens={c:i for i,c in enumerate(alphabet)}
sequence=np.array([tokens[c] for c in 'abcabcabcabc'])
inputs=np.eye(3)[sequence[:-1]]; targets=sequence[1:]
Wx=rng.normal(0,.2,(3,6)); Wh=rng.normal(0,.2,(6,6)); Wy=rng.normal(0,.2,(6,3))
b=np.zeros(6); c=np.zeros(3)
def softmax(z):
    ex=np.exp(z-z.max()); return ex/ex.sum()
def forward(xs):
    hs=[np.zeros(6)]; ps=[]
    for x in xs:
        hs.append(np.tanh(x@Wx+hs[-1]@Wh+b)); ps.append(softmax(hs[-1]@Wy+c))
    return hs,np.array(ps)
print('States/output:',len(forward(inputs)[0]),forward(inputs)[1].shape)'''),
M('''## BPTT: error dari masa depan
Gradien hidden pada t menerima dua kontribusi: loss output pada t dan state berikutnya melalui W_h. Kalikan jumlahnya dengan 1−h_t², lalu akumulasi gradien setiap matriks yang dipakai ulang. Gradien dibagi jumlah timestep karena loss dirata-ratakan. Gradient clipping membatasi norm pembaruan ketika gradien membesar; clipping tidak memperbaiki vanishing gradient.'''),
C('''def gradients(xs,ys):
    hs,ps=forward(xs)
    gWx=np.zeros_like(Wx); gWh=np.zeros_like(Wh); gWy=np.zeros_like(Wy)
    gb=np.zeros_like(b); gc=np.zeros_like(c); future=np.zeros(6)
    for t in reversed(range(len(xs))):
        dz=ps[t].copy(); dz[ys[t]]-=1; dz/=len(xs)
        gWy+=np.outer(hs[t+1],dz); gc+=dz
        dh=(dz@Wy.T+future)*(1-hs[t+1]**2)
        gWx+=np.outer(xs[t],dh); gWh+=np.outer(hs[t],dh); gb+=dh
        future=dh@Wh.T
    loss=-np.log(ps[np.arange(len(ys)),ys]+1e-12).mean()
    return loss,[gWx,gWh,gWy,gb,gc]
loss,g=gradients(inputs,targets)
original=Wh[0,0]; eps=1e-5
Wh[0,0]=original+eps; plus=gradients(inputs,targets)[0]
Wh[0,0]=original-eps; minus=gradients(inputs,targets)[0]
Wh[0,0]=original
print('Wh gradient analytic/numeric:',g[1][0,0],(plus-minus)/(2*eps))
assert np.isclose(g[1][0,0],(plus-minus)/(2*eps),atol=1e-6)
losses=[]
for epoch in range(300):
    loss,gs=gradients(inputs,targets); losses.append(loss)
    norm=np.sqrt(sum(np.sum(a*a) for a in gs)); scale=min(1,1/(norm+1e-12))
    for parameter,grad in zip([Wx,Wh,Wy,b,c],gs): parameter-=.3*scale*grad
print('Loss awal/akhir:',losses[0],losses[-1])
plt.plot(losses); plt.xlabel('Epoch'); plt.ylabel('Cross-entropy'); plt.show()
assert losses[-1]<losses[0]'''),
C('''h=np.zeros(6); token=tokens['a']; generated='a'
for _ in range(20):
    h=np.tanh(np.eye(3)[token]@Wx+h@Wh+b)
    token=int(np.argmax(softmax(h@Wy+c))); generated+=alphabet[token]
print('Greedy generation:',generated)'''),
M('''## Ringkasan
State menyediakan representasi untuk urutan panjang bervariasi. BPTT menjumlahkan kontribusi bobot dari semua timestep, dan finite difference memeriksa implementasinya. Teks periodik pendek mudah dihafal; hasil generasi ini bukan bukti pemahaman bahasa. Urutan panjang meningkatkan biaya penyimpanan state dan risiko vanishing/exploding gradients, yang memotivasi LSTM dan truncated BPTT.''')])

write(13, 'Deep_Learning_Framework', [
M('''# Bab 13 — Let’s Build a Deep Learning Framework
**Acuan:** Trask, *Grokking Deep Learning*, Bab 13 (hlm. 231–263).

Tujuan: membangun computational graph, reverse-mode automatic differentiation, dan optimizer sederhana. Buku membangun tensor/layer termasuk embedding, cross-entropy, dan recurrent layer; contoh mandiri ini membatasi mesin autograd pada skalar agar graph bercabang dan akumulasi gradien terlihat jelas. Operasi vektor dan recurrent telah dihitung manual di bab sebelumnya.'''),
M('''## Graph dan chain rule
Setiap hasil operasi menyimpan parent dan aturan backward. Reverse-mode menyusun node secara topologis lalu berjalan terbalik, sehingga semua jalur menuju parent berkontribusi sebelum parent diproses. Gradien harus ditambah, bukan ditimpa: pada z=x*x+x, x dipakai tiga kali. Ini bukan symbolic differentiation maupun finite difference; setiap operasi lokal menyediakan turunan analitik.'''),
C('''import math
class Value:
    def __init__(self,data,parents=()):
        self.data=float(data); self.grad=0.; self.parents=parents; self._backward=lambda:None
    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other))
        def backward():
            self.grad+=out.grad; other.grad+=out.grad
        out._backward=backward; return out
    __radd__=__add__
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other))
        def backward():
            self.grad+=other.data*out.grad; other.grad+=self.data*out.grad
        out._backward=backward; return out
    __rmul__=__mul__
    def tanh(self):
        out=Value(math.tanh(self.data),(self,))
        def backward(): self.grad+=(1-out.data**2)*out.grad
        out._backward=backward; return out
    def backward(self):
        ordered=[]; visited=set()
        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents: visit(parent)
                ordered.append(node)
        visit(self)
        for node in ordered: node.grad=0.
        self.grad=1.
        for node in reversed(ordered): node._backward()

x=Value(3); z=x*x+x; z.backward()
print('z:',z.data,'dz/dx:',x.grad)
assert z.data==12 and x.grad==7
a=Value(.4); out=(a*a+a).tanh(); out.backward()
eps=1e-6; f=lambda v:math.tanh(v*v+v)
assert abs(a.grad-(f(.4+eps)-f(.4-eps))/(2*eps))<1e-6'''),
M('''## Parameter, layer, loss, optimizer
Layer mengelola parameter dan forward; loss mengubah prediksi menjadi skalar; optimizer memperbarui parameter dari gradien. Pemisahan ini memungkinkan model berganti tanpa menulis ulang aturan chain rule. Graph baru dibangun tiap iterasi. Implementasi ini mereset gradien graph saat backward; framework besar memiliki aturan akumulasi yang lebih fleksibel dan biasanya meminta `zero_grad` secara eksplisit.'''),
C('''class Linear:
    def __init__(self): self.w=Value(.1); self.b=Value(0.)
    def __call__(self,x): return self.w*x+self.b
    def parameters(self): return [self.w,self.b]
class SGD:
    def __init__(self,parameters,lr): self.parameters=parameters; self.lr=lr
    def step(self):
        for p in self.parameters: p.data-=self.lr*p.grad
model=Linear(); optimizer=SGD(model.parameters(),.05)
data=[(-1.,-1.),(0.,1.),(1.,3.),(2.,5.)]
history=[]
for epoch in range(150):
    errors=[model(x)+(-target) for x,target in data]
    loss=sum(error*error for error in errors)*(1/len(data))
    loss.backward(); optimizer.step(); history.append(loss.data)
print('w,b,loss:',model.w.data,model.b.data,history[-1])
assert history[-1]<1e-5'''),
M('''## Perluasan dan ringkasan
Tensor memerlukan broadcasting beserta reduksi gradien balik, matrix multiplication, indexing dengan akumulasi untuk indeks berulang, dan fungsi loss stabil. Recurrent layer memerlukan parameter bersama pada banyak timestep—uji graph bercabang di atas adalah dasar pentingnya. Mesin kecil ini memeriksa chain rule dan optimizer; tidak mengklaim kelengkapan, kecepatan, GPU, atau pengganti framework produksi.''')])

write(14, 'LSTM', [
M('''# Bab 14 — Long Short-Term Memory
**Acuan:** Trask, *Grokking Deep Learning*, Bab 14 (hlm. 265–280).

Tujuan: memahami cell state, gates, masalah gradien, truncated BPTT, dan character language modeling. Korpus periodik buatan sendiri menggantikan Shakespeare. Agar semua komputasi tetap mudah ditelusuri, optimasi LSTM mini memakai finite difference; ini adaptasi pedagogis yang jauh lebih lambat daripada autograd/BPTT buku.'''),
M('''## Memory dan gates
LSTM memiliki forget gate f, input gate i, kandidat g, dan output gate o. Cell state diperbarui c_t=f_t*c_(t−1)+i_t*g_t, lalu h_t=o_t*tanh(c_t). Sigmoid membatasi gates ke (0,1), sedangkan tanh membatasi kandidat. Jalur aditif cell membantu menjaga informasi; ia tidak menjamin semua masalah vanishing gradient hilang. Turunan lokal terhadap cell sebelumnya sepanjang jalur langsung adalah f_t, sehingga gate mendekati satu mempertahankan sinyal lebih lama.'''),
C('''import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(14)
sigmoid=lambda x:1/(1+np.exp(-np.clip(x,-30,30)))
alphabet='ab'; hidden=2; width=2+hidden
W=rng.normal(0,.2,(width,4*hidden)); bias=np.zeros(4*hidden)
V=rng.normal(0,.2,(hidden,2)); output_bias=np.zeros(2)
parameters=[W,bias,V,output_bias]
def step(token,h,c):
    joined=np.r_[np.eye(2)[token],h]
    z=joined@W+bias
    f=sigmoid(z[:2]); i=sigmoid(z[2:4]); g=np.tanh(z[4:6]); o=sigmoid(z[6:])
    c=f*c+i*g; h=o*np.tanh(c)
    logits=h@V+output_bias; p=np.exp(logits-logits.max()); p/=p.sum()
    return h,c,p
tokens=np.array([0,1,0,1,0,1,0])
def chunk_loss():
    h=np.zeros(hidden); c=np.zeros(hidden); loss=0.
    for token,target in zip(tokens[:-1],tokens[1:]):
        h,c,p=step(token,h,c); loss-=np.log(p[target]+1e-12)
    return loss/(len(tokens)-1)
print('Jumlah parameter:',sum(p.size for p in parameters),'loss awal:',chunk_loss())'''),
M('''## Training mini dan batas finite difference
Turunan dL/dθ diperkirakan dengan [L(θ+ε)−L(θ−ε)]/(2ε). Setiap parameter memerlukan dua forward pass sehingga metode ini hanya layak untuk contoh sangat kecil; LSTM praktis memakai reverse-mode autograd dan BPTT. Contoh mengoptimalkan satu chunk enam transisi dan mereset state pada awal chunk. Truncated BPTT umumnya membawa nilai state ke chunk berikutnya sambil memutus graph gradien pada batas chunk; mereset state seperti di sini adalah kebijakan yang berbeda dan menghilangkan konteks lintas chunk.'''),
C('''losses=[]; epsilon=1e-4
for epoch in range(100):
    losses.append(chunk_loss()); grads=[]
    for parameter in parameters:
        grad=np.zeros_like(parameter)
        for idx in np.ndindex(parameter.shape):
            value=parameter[idx]
            parameter[idx]=value+epsilon; plus=chunk_loss()
            parameter[idx]=value-epsilon; minus=chunk_loss()
            parameter[idx]=value
            grad[idx]=(plus-minus)/(2*epsilon)
        grads.append(grad)
    for parameter,grad in zip(parameters,grads): parameter-=.8*grad
print('Loss awal/akhir:',losses[0],chunk_loss())
plt.plot(losses); plt.xlabel('Epoch'); plt.ylabel('Cross-entropy'); plt.show()
assert chunk_loss()<losses[0]
h=np.zeros(hidden); cell=np.zeros(hidden); token=0; generated='a'
for _ in range(20):
    h,cell,p=step(token,h,cell); token=int(p.argmax()); generated+=alphabet[token]
print('Greedy sample:',generated)'''),
M('''## Sampling, truncated BPTT, dan ringkasan
Greedy sampling memilih kelas terbesar; stochastic sampling dari softmax dengan temperature positif memberi variasi. Temperature rendah menajamkan distribusi, tinggi meratakannya. Pelatihan teks nyata membutuhkan korpus besar, urutan validasi terpisah, batch/chunk, pemotongan norm gradien, dan optimasi efisien. LSTM memperkenalkan kontrol memori melalui gates; penurunan loss pada pola dua karakter hanya membuktikan mekanisme mini ini berjalan, bukan kemampuan menulis bahasa alami.''')])

write(15, 'Federated_Learning', [
M('''# Bab 15 — Deep Learning on Unseen Data: Federated Learning
**Acuan:** Trask, *Grokking Deep Learning*, Bab 15 (hlm. 281–291).

Tujuan: melatih model pada beberapa client, menggabungkan pembaruan, memahami kebocoran gradien, dan membedakan secure aggregation dari enkripsi homomorfik. Data spam mini dibuat sendiri dan seluruh client disimulasikan dalam satu proses; ini tidak menerapkan jaringan federasi atau kriptografi produksi.'''),
M('''## Model berpindah ke data
Server mengirim bobot global; client melakukan langkah training lokal; server menggabungkan bobot client dengan rata-rata berbobot jumlah contoh (FedAvg). Tujuan global adalah jumlah loss lokal berbobot ukuran dataset. Data non-IID, jumlah langkah lokal, client yang absen, dan ukuran dataset yang tidak sama dapat mengubah konvergensi. Data yang tetap lokal mengurangi kebutuhan sentralisasi, tetapi pembaruan model juga dapat membocorkan informasi.'''),
C('''import numpy as np
import matplotlib.pyplot as plt
vocab=['free','prize','meeting','project']
# Bag-of-words: contoh sintetis, bukan email pengguna.
clients=[(np.array([[1.,1.,0.,0.],[0.,0.,1.,1.],[1.,0.,0.,0.]]),np.array([1.,0.,1.])),
         (np.array([[0.,1.,0.,0.],[0.,0.,1.,0.]]),np.array([1.,0.])),
         (np.array([[1.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,1.],[0.,1.,0.,0.]]),np.array([1.,0.,0.,1.]))]
sigmoid=lambda z:1/(1+np.exp(-np.clip(z,-30,30)))
def local_train(weights,X,y,steps=3,lr=.3):
    w=weights.copy()
    for _ in range(steps): w-=lr*(X.T@(sigmoid(X@w)-y)/len(y))
    return w
def global_loss(w):
    X=np.vstack([x for x,y in clients]); y=np.concatenate([y for x,y in clients])
    p=sigmoid(X@w)
    return -np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12))
w=np.zeros(4); history=[]
for round_id in range(60):
    history.append(global_loss(w))
    updates=[local_train(w,X,y) for X,y in clients]
    sizes=np.array([len(y) for X,y in clients])
    w=np.average(updates,axis=0,weights=sizes)
print('Loss awal/akhir:',history[0],global_loss(w))
print('Bobot:',dict(zip(vocab,w.round(3))))
plt.plot(history); plt.xlabel('Federated round'); plt.ylabel('Global BCE'); plt.show()
assert global_loss(w)<history[0]'''),
M('''## Mengapa update bisa membocorkan data?
Untuk satu contoh pada model logistic dengan bobot nol, gradien adalah (0,5−y)x. Pada label spam y=1, koordinat nonzero pada gradien langsung mengungkap token yang muncul. Serangan ini adalah ilustrasi aljabar pada data sintetis sendiri; agregasi banyak contoh dan banyak langkah mengubah kemudahan rekonstruksi, tetapi tidak memberi jaminan privasi dengan sendirinya.'''),
C('''toy_x=np.array([1.,1.,0.,0.]); toy_y=1.
gradient=(.5-toy_y)*toy_x
recovered=gradient/(.5-toy_y)
print('Gradient:',gradient,'recovered presence:',recovered)
assert np.allclose(recovered,toy_x)'''),
M('''## Secure aggregation: ilustrasi pembatalan mask
Client dapat menambah mask berpasangan sehingga mask saling menghapus pada jumlah akhir. Server melihat jumlah update, bukan update individual dalam model ancaman dan protokol yang sesuai. Contoh berikut mengilustrasikan identitas penjumlahannya saja: semua mask diketahui satu proses, tanpa pertukaran kunci, penanganan dropout client, atau perlindungan dari kolusi. Enkripsi homomorfik berbeda: operasi tertentu dilakukan pada ciphertext dan hasil didekripsi oleh pemegang kunci; tidak diimplementasikan oleh mask sederhana ini.'''),
C('''rng=np.random.default_rng(15)
updates=np.array([[.1,.2],[-.2,.3],[.4,-.1]])
mask01=rng.normal(size=2); mask02=rng.normal(size=2); mask12=rng.normal(size=2)
masked=np.array([updates[0]+mask01+mask02,
                 updates[1]-mask01+mask12,
                 updates[2]-mask02-mask12])
print('Agregat bermask:',masked.sum(0),'agregat asli:',updates.sum(0))
assert np.allclose(masked.sum(0),updates.sum(0))'''),
M('''## Ringkasan dan keterbatasan
FedAvg menggabungkan hasil training lokal; rata-rata berbobot penting saat ukuran client berbeda. Federasi sendiri bukan jaminan privasi. Secure aggregation melindungi visibilitas update individual pada asumsi tertentu, sementara differential privacy dapat membatasi pengaruh individu dengan biaya utilitas. Enkripsi homomorfik menyediakan operasi pada data terenkripsi dengan biaya komputasi dan pengelolaan kunci. Penerapan nyata memerlukan protokol teruji serta evaluasi model ancaman; simulasi ini hanya menjelaskan mekanismenya.''')])
