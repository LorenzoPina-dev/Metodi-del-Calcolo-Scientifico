# Matrix TS – Linear Algebra Library in TypeScript

![Matrix Logo](https://via.placeholder.com/200x80?text=Matrix+TS)

Una libreria TypeScript per algebra lineare avanzata: decomposizioni LU/LUP, Cholesky, risoluzione di sistemi lineari, determinante, inversa, operazioni su matrici dense e sparse, con attenzione a stabilità numerica e performance.

---

## 📚 Teoria

### 1. Sistemi lineari

Un sistema lineare può essere scritto come:

\[
A x = b
\]

dove:

- \(A \in \mathbb{R}^{n \times n}\) è la matrice dei coefficienti,
- \(x \in \mathbb{R}^{n}\) è il vettore delle incognite,
- \(b \in \mathbb{R}^{n}\) è il termine noto.

---

### 2. Decomposizioni matriciali

#### 2.1 LU senza pivoting

\[
A = L U
\]

- \(L\) è una matrice triangolare inferiore con diagonale 1,
- \(U\) è una matrice triangolare superiore.

Vantaggi:

- Risoluzione rapida dei sistemi lineari con più vettori \(b\).
- Calcolo determinante: \(\det(A) = \prod_i U_{ii}\).

Limitazioni:

- Instabile se \(A\) contiene elementi piccoli sulla diagonale (pivoting richiesto).

---

#### 2.2 LUP (LU con pivoting)

\[
P A = L U
\]

- \(P\) è una matrice di permutazione,
- migliora la stabilità numerica rispetto alla LU semplice.

**Pivoting** evita divisioni per numeri troppo piccoli e riduce l’errore numerico.

---

#### 2.3 Cholesky

Per matrici simmetriche e positive definite:

\[
A = L L^T
\]

- Soluzione dei sistemi: due sostituzioni triangolari (forward e backward).
- Più efficiente di LU per matrici SPD.

---

### 3. Risoluzione sistemi

Per sistemi triangolari \(L x = b\) o \(U x = b\):

- **Forward substitution** (per L)  
\[
x_i = \frac{1}{L_{ii}}\left(b_i - \sum_{j=1}^{i-1} L_{ij} x_j\right)
\]

- **Backward substitution** (per U)  
\[
x_i = \frac{1}{U_{ii}}\left(b_i - \sum_{j=i+1}^{n} U_{ij} x_j\right)
\]

La libreria implementa metodi ottimizzati per matrici multiple colonne (multirhs).

---

### 4. Determinante e inversa

- Determinante tramite LU/LUP:

\[
\det(A) = \text{sign(P)} \prod_i U_{ii}
\]

- Inversa tramite risoluzione sistematica con \(I\) come \(b\):

\[
A^{-1} = \text{solve}(A, I)
\]

---

### 5. Matrici sparse

- È possibile creare matrici quasi triangolari o con diagonali predominanti.
- La libreria può sfruttare la struttura per velocizzare decomposizioni e soluzioni.

---

## ⚙️ Installazione

```bash
npm install matrix-ts
```

Oppure clona il repository:

```bash
git clone https://github.com/tuonome/matrix-ts.git
cd matrix-ts
npm install
```

---

## 🧪 Utilizzo Base

```ts
import { Matrix } from "./Matrix";

const A = new Matrix(3,3);
A.set(0,0,2); A.set(0,1,1); A.set(0,2,3);
A.set(1,0,4); A.set(1,1,1); A.set(1,2,6);
A.set(2,0,7); A.set(2,1,8); A.set(2,2,9);

const b = new Matrix(3,1);
b.set(0,0,1); b.set(1,0,2); b.set(2,0,3);

// Risoluzione sistema lineare
const x = A.solve(b);
console.log(x.toString());

// Decomposizione LUP
const {L, U, P} = A.lup();
console.log(L.toString(), U.toString(), P);
```

---

## 📊 Benchmark e Visualizzazioni

La libreria include script di benchmark per valutare:

- Tempi di decomposizione LU/LUP/Cholesky,
- Tempi di soluzione sistemi lineari,
- Determinante e inversa,
- Stabilità numerica su matrici con condizionamento elevato,
- Sparse vs Dense performance.

Esempio di visualizzazione con Plotly (TS):

```ts
import Plotly from "plotly.js-dist";

const sizes = [5, 50, 100, 200, 500, 1000];
const timesLU = [/* tempi raccolti */];

const trace = {
  x: sizes,
  y: timesLU,
  type: "scatter",
  mode: "lines+markers",
  name: "LU"
};

Plotly.newPlot("plotDiv", [trace], {title: "Benchmark LU"});
```

---

### ✅ Tipi di test inclusi

1. **Correttezza**: confronto con soluzioni note o con librerie di riferimento.
2. **Velocità**: matrici da 5x5 fino a 1000x1000.
3. **Stabilità numerica**: test su matrici con condizionamento elevato.
4. **Sparse matrices**: valutazione della performance su diagonali predominanti.

---

## 📈 Risultati attesi

- Decomposizione LU/LUP stabile fino a 1000x1000.
- Cholesky più veloce per SPD.
- Forward/backward substitution ottimizzata per multirhs.
- Tempi aumentano circa come \(O(n^3)\) per decomposizione, \(O(n^2 m)\) per sistemi con \(m\) colonne.

---

## 📂 Struttura progetto

```
matrix-ts/
│
├─ src/
│  ├─ Matrix.ts       # classe principale
│  ├─ benchmark.ts    # script benchmark
│  └─ tests/
│     ├─ test_solve.ts
│     ├─ test_decomp.ts
│     └─ test_speed.ts
│
├─ package.json
└─ README.md
```

---

## 📝 Note

- Tutti i metodi supportano **broadcasting** per add/subtract tra righe/colonne.
- Stabilità e performance testate su matrici casuali, diagonali predominanti, e matrici sparse.
- È consigliabile usare LUP su matrici generiche per garantire stabilità.