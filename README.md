# 🧮 Matrix — Libreria TypeScript di Algebra Lineare

Una libreria **TypeScript** per il calcolo scientifico, ispirata all'API di MATLAB/NumPy.
Ottimizzata con `Float64Array`, loop unrolling (×4) e ordinamento cache-friendly `i-k-j` per massimizzare le prestazioni in ambiente JavaScript/Node.js.

---

## 📁 Struttura del Progetto

```
typescript/
├── src/
│   ├── Matrix.ts              # Classe principale (entry point)
│   ├── index.ts               # Export pubblico
│   ├── core/
│   │   └── MatrixBase.ts      # Classe base: storage, get/set, clone, toString
│   ├── ops/                   # Operazioni matematiche
│   │   ├── add.ts             # Addizione con broadcasting
│   │   ├── subtract.ts        # Sottrazione con broadcasting
│   │   ├── multiply.ts        # Moltiplicazione matriciale (i-k-j cache-friendly)
│   │   ├── dotOps.ts          # Operazioni element-wise (dotMul, dotDiv, dotPow)
│   │   ├── det.ts             # Determinante (multi-strategia)
│   │   ├── norm.ts            # Norme vettoriali e matriciali
│   │   ├── pow.ts             # Potenza matriciale (esponenziazione rapida)
│   │   ├── statistics.ts      # sum, mean, max, min per righe/colonne
│   │   ├── transform.ts       # transpose, reshape, flip, rot90, repmat, slice, circshift
│   │   ├── unary.ts           # abs, sqrt, round, floor, ceil, exp, sin, cos, tan
│   │   ├── equal.ts           # Confronto con tolleranza
│   │   └── hasProperty.ts     # Predicati strutturali (isSymmetric, isPositiveDefinite, ...)
│   ├── init/                  # Costruttori e generatori
│   │   ├── init.ts            # zeros, ones, identity, diag, diagFromArray
│   │   ├── random.ts          # Generazione casuale (uniforme, normale, intera)
│   │   ├── sparse.ts          # Matrici sparse (COO, random density, diagonale)
│   │   ├── structured.ts      # toeplitz, hankel, vander
│   │   └── known/             # Galleria di matrici note (24 matrici)
│   ├── decomposition/         # Fattorizzazioni
│   │   ├── lu.ts              # LU (eliminazione gaussiana senza pivot)
│   │   ├── lup.ts             # LUP (con pivoting parziale)
│   │   ├── lu_total.ts        # LU con pivot totale (ritorna P come Matrix)
│   │   ├── cholesky.ts        # Cholesky (solo per SPD)
│   │   ├── qr.ts              # QR (Gram-Schmidt classico)
│   │   ├── dlu.ts             # Scomposizione additiva A = D + L + U
│   │   ├── tril.ts            # Estrazione triangolare inferiore
│   │   └── triu.ts            # Estrazione triangolare superiore
│   ├── solver/                # Sistemi lineari Ax = b
│   │   ├── solve.ts           # Dispatcher: LU, LUP, CHOLESKY, QR, JACOBI, GAUSS-SEIDEL
│   │   ├── triangular.ts      # Sostituzione in avanti/indietro
│   │   ├── jacobi.ts          # Metodo di Jacobi (iterativo matriciale)
│   │   ├── gausSeidelMat.ts   # Metodo di Gauss-Seidel (iterativo matriciale)
│   │   └── _hasConverged.ts   # Criterio di arresto (norma infinito)
│   └── algoritm/
│       └── inverse.ts         # smartInverse: inversione adattiva multi-strategia
├── test/
│   ├── Matrix.test.ts
│   ├── Matrix.robust.test.ts
│   ├── Scomposizioni.test.ts
│   ├── Matrix.benchmark.ts    # Benchmark completo (tutti i metodi, 6 dimensioni)
│   └── confronti.benchmark.ts # Benchmark comparativo dei solver
├── matrix_full_benchmark_results.json
└── package.json
```

---

## ⚙️ Installazione e Avvio

```bash
# Clona il repository
cd typescript

# Installa le dipendenze
npm install

# Esegui i test
npm test

# Compila in JavaScript
npx tsc

# Esegui un benchmark
npx ts-node test/Matrix.benchmark.ts
```

**Dipendenze principali:** `vitest` (testing), `plotly.js` (visualizzazione), `jsdom`.

---

## 🏗️ Architettura Interna

### `MatrixBase` — il fondamento

`MatrixBase` è la classe base da cui `Matrix` eredita. Gestisce lo storage, l'indicizzazione e le primitive fondamentali.

```typescript
class MatrixBase {
    data: Float64Array;  // Buffer contiguo row-major
    rows: number;
    cols: number;
}
```

**Perché `Float64Array`?**
- Allocazione contigua in memoria → accessi cache-friendly
- Aritmetica a 64-bit (doppia precisione, identica ai `double` di C)
- Nessun overhead di boxing/unboxing rispetto ai `number[]` normali

**Indicizzazione row-major:**
```
indice = i * cols + j
```

**Costanti:**
```typescript
static EPS = 1e-10   // Soglia numerica per confronti
static isZero(x)     // |x| < EPS
```

**Metodi base:**
| Metodo | Descrizione |
|--------|-------------|
| `get(i, j)` | Legge l'elemento alla riga `i`, colonna `j` |
| `set(i, j, v)` | Scrive il valore `v` in posizione `(i, j)` |
| `clone()` | Copia profonda (nuovo `Float64Array`) |
| `toString()` | Stampa la matrice formattata con 3 decimali |

---

## 📐 Classe `Matrix`

`Matrix` estende `MatrixBase` aggiungendo tutte le operazioni di alto livello.
È l'unica classe da importare negli utilizzi normali.

```typescript
import { Matrix } from './src';
```

---

## 🔨 Costruttori e Factory

### Matrici standard

```typescript
Matrix.zeros(rows, cols)          // Matrice di zeri
Matrix.ones(rows, cols)           // Matrice di uni
Matrix.identity(n)                // Matrice identità n×n
Matrix.diag(n, k)                 // Diagonale con valore k
Matrix.diagFromArray([1, 2, 3])   // Diagonale da array
```

### Matrici casuali

```typescript
Matrix.random(n, m)
// Opzioni disponibili:
Matrix.random(n, m, { type: 'uniform', min: 0, max: 1 })   // default
Matrix.random(n, m, { type: 'normal',  mean: 0, std: 1 })  // Box-Muller
Matrix.random(n, m, { type: 'int',     min: 1, max: 10 })  // Interi
```

### Matrici sparse

```typescript
// COO (Coordinate Format)
Matrix.sparse(n, m, { type: 'coo', rows: [0,1], cols: [0,1], values: [3,5] })

// Densità random
Matrix.sparse(n, m, { type: 'random', density: 0.1, min: 0, max: 1 })

// Diagonale spostata
Matrix.sparse(n, m, { type: 'diag', values: [1,2,3], k: 1 })
```

### Matrici strutturate

```typescript
Matrix.toeplitz(c, r?)    // Matrice di Toeplitz (c = prima colonna, r = prima riga)
Matrix.hankel(c, r?)      // Matrice di Hankel
Matrix.vander([1,2,3])    // Matrice di Vandermonde
Matrix.tril(A)            // Triangolare inferiore di A
Matrix.triu(A)            // Triangolare superiore di A (con offset k)
```

---

## 🖼️ Galleria di Matrici Note (`Matrix.gallery`)

Raccolta di 24 matrici classiche per il testing di algoritmi numerici.

### Matrici Classiche e Strutturate

| Funzione | Dimensione | Proprietà principali |
|----------|-----------|----------------------|
| `hilbert(n)` | n×n | Simmetrica, SPD, **altamente mal condizionata** — `H(i,j)=1/(i+j-1)` |
| `pascal(n)` | n×n | Simmetrica, SPD, `det=1`, coefficienti binomiali |
| `magic(n)` | n×n | Somme righe/colonne/diagonali uguali, costante = `n(n²+1)/2` |
| `lehmer(n)` | n×n | Simmetrica, SPD, `A(i,j)=min(i,j)/max(i,j)`, inversa tridiagonale |
| `minij(n)` | n×n | `A(i,j)=min(i,j)`, definita positiva |
| `circul(v)` | n×n | Circolante dalla riga `v` |
| `tridiag(a,b,c)` | n×n | Tridiagonale da tre array (sotto/principale/sopra) |

### Matrici di Test e Analisi

| Funzione | Proprietà / Uso tipico |
|----------|------------------------|
| `cauchy(x,y?)` | `C(i,j)=1/(x_i+y_j)`, test di inversione |
| `frank(n)` | Autovalori tutti positivi, utile per test QR |
| `grcar(n,k?)` | Non simmetrica, autovalori complessi vicini al cerchio unitario |
| `kahan(n)` | Mal condizionata, testa l'accuratezza della fattorizzazione QR |
| `invhess(n)` | Inversa di una matrice di Hessenberg |
| `binomial(n)` | Coefficienti binomiali, autovalori = `2^k` |
| `wilkinson(n)` | Matrice di Wilkinson, test classico per autovalori |

### Matrici per PDE e Elementi Finiti

| Funzione | Descrizione |
|----------|-------------|
| `wathen(nx, ny)` | Matrice sparsa SPD da elementi finiti isoparametrici 8 nodi. Dimensione: `3*nx*ny + 2*nx + 2*ny + 1` |
| `neumann(n)` | Matrice del Laplaciano con condizioni di Neumann |
| `dorr(n)` | Matrice tridiagonale con convezione-diffusione dominata |

### Matrici di Trasformazione e Ortogonali

| Funzione | Descrizione |
|----------|-------------|
| `house(x)` | Riflettore di Householder |
| `orthog(n)` | Matrice ortogonale (Q^T Q = I) |
| `randsvd(n,k,mode)` | Matrice con valori singolari prescritti |

### Matrici con Proprietà Spettrali

| Funzione | Descrizione |
|----------|-------------|
| `fiedler(c)` | Matrice di Fiedler, simmetrica |
| `hanowa(n)` | Matrice a blocchi 2×2 con autovalori immaginari puri |
| `smoke(n)` | Autovalori su una spirale nel piano complesso |

**Utilizzo:**
```typescript
const H = Matrix.gallery.hilbert(6);
const W = Matrix.gallery.wathen(3, 3);
const M = Matrix.gallery.magic(5);
```

---

## ➕ Operazioni Aritmetiche

### Addizione e Sottrazione

Entrambe supportano **broadcasting** MATLAB-style:

```typescript
A.add(B)    // A + B
A.sub(B)    // A - B
```

| Forma di `B` | Comportamento |
|--------------|---------------|
| `number` | Scalare aggiunto a ogni elemento |
| `Matrix` stessa dimensione | Somma elemento per elemento |
| `Matrix` 1×N (vettore riga) | Broadcasting su ogni riga |
| `Matrix` M×1 (vettore colonna) | Broadcasting su ogni colonna |

**Ottimizzazione:** loop unrolling ×4 su buffer contiguo.

### Moltiplicazione Matriciale

```typescript
A.mul(B)      // Prodotto matriciale (A*B) — richiede A.cols == B.rows
A.mul(3.14)   // Prodotto per scalare
```

**Algoritmo:** ordine `i-k-j` (cache-friendly) con loop unrolling ×4 sulle colonne. Il valore `A[i,k]` viene caricato una sola volta in un registro e riutilizzato per aggiornare un'intera riga di output, massimizzando i cache hit su `B`.

Complessità: O(M·K·N).

### Potenza Matriciale

```typescript
A.pow(8)   // Esponenziazione rapida (binary exponentiation)
```

Usa l'algoritmo di **esponenziazione rapida** (square-and-multiply): complessità O(n³ log k) invece di O(n³ · k). Solo esponenti interi non negativi.

### Operazioni Element-wise (dot)

```typescript
A.dotMul(B)    // A .* B   (prodotto di Hadamard)
A.dotDiv(B)    // A ./ B
A.dotPow(2)    // A .^ 2   (ogni elemento elevato al quadrato)
```

Supportano broadcasting (stessa riga/colonna vettore) come `add`.

---

## 📊 Algebra Lineare

### Determinante

```typescript
A.det()
```

**Strategia adattiva** (4 casi):
1. **1×1**: valore diretto
2. **2×2**: formula `ad-bc`
3. **Triangolare o Diagonale**: prodotto della diagonale — O(n)
4. **Caso generale**: decomposizione LUP → `det = (-1)^swaps * prod(diag(U))` — O(n³)

### Norma

```typescript
A.norm(1)      // Norma 1: max somma assoluta delle colonne
A.norm(2)      // Norma 2: valore singolare massimo (richiede SVD)
A.norm('inf')  // Norma infinito: max somma assoluta delle righe
A.norm('fro')  // Norma di Frobenius: sqrt(sum(aij²))
```

Per vettori (righe o colonne), `norm(2)` e `norm('fro')` coincidono.

### Traccia

```typescript
A.trace()   // Somma degli elementi diagonali
```

Ottimizzata con `stride = cols + 1` (accesso diretto al buffer senza doppio loop).

### Inversione (`smartInverse`)

```typescript
A.inv()       // equivalente a A.inverse()
```

**`smartInverse`** sceglie automaticamente l'algoritmo più efficiente analizzando la struttura della matrice:

```
A rettangolare?     → Pseudo-inversa di Moore-Penrose: (A^T A)^{-1} A^T
A diagonale?        → Reciproco della diagonale: O(n)
A triangolare sup.? → Sostituzione indietro colonna per colonna: O(n²)
A triangolare inf.? → Sostituzione avanti colonna per colonna: O(n²)
A ortogonale?       → Trasposta: A^{-1} = A^T (O(n²))
Altrimenti          → Decomposizione LUP + sostituzione: O(n³)
```

### Uguaglianza con Tolleranza

```typescript
A.equals(B)            // con tolleranza default EPS = 1e-10
A.equals(B, 1e-6)      // con tolleranza personalizzata
```

---

## 📉 Statistiche

```typescript
A.sum(1)    // Somma colonne  → vettore 1×N
A.sum(2)    // Somma righe    → vettore M×1
A.mean(1)   // Media colonne
A.mean(2)   // Media righe
A.max()     // { value: Matrix, index: Int32Array }  (indici 1-based)
A.min()     // { value: Matrix, index: Int32Array }
A.totalSum() // Somma scalare di tutti gli elementi
```

`dim=1` → riduzione verticale (per colonne), `dim=2` → riduzione orizzontale (per righe).
Stessa semantica di `sum(A, 1)` e `sum(A, 2)` in MATLAB.

---

## 🔄 Trasformazioni

```typescript
A.t()                             // Trasposta
A.reshape(r, c)                   // Cambia forma (condivide il buffer — O(1)!)
A.slice(rowStart, rowEnd, colStart, colEnd)  // Sottomatrice
A.repmat(r, c)                    // Replica r×c volte (come kron(ones(r,c), A))
A.flip(1)                         // Ribalta verticalmente (flipud)
A.flip(2)                         // Ribalta orizzontalmente (fliplr)
A.rot90(k)                        // Rotazione 90°×k in senso antiorario (k=1,2,3)
```

> `reshape` è **O(1)** perché condivide il buffer `Float64Array` originale (zero-copy).

---

## 🔢 Operazioni Unarie (element-wise)

```typescript
A.abs()     // |aij|
A.sqrt()    // √aij
A.round()   // Arrotondamento
// Disponibili nel modulo ma non esposte sulla classe:
// exp, floor, ceil, sin, cos, tan
```

Tutte usano il pattern `applyUnary` con loop unrolling ×4.

---

## 🔍 Predicati Strutturali

Tutti i metodi seguenti restituiscono `boolean` e usano una soglia di tolleranza (`tol = 1e-10` di default).

### Proprietà Geometriche

```typescript
A.isSquare()             // rows === cols
A.isSymmetric(tol?)      // A[i,j] ≈ A[j,i]
A.isUpperTriangular(tol?) // tutti i sub-diagonali ≈ 0
A.isLowerTriangular(tol?) // tutti i sup-diagonali ≈ 0
A.isDiagonal(tol?)        // tutti i fuori-diagonali ≈ 0
A.isIdentity(tol?)        // diagonale = 1, resto = 0
A.isOrthogonal(tol?)      // A^T A ≈ I (confronta colonne)
A.isZeroMatrix(tol?)      // tutti gli elementi ≈ 0
```

### Proprietà Algebriche Avanzate

```typescript
A.isInvertible(tol?)       // det(U) di LUP > tol
A.isSingular(tol?)         // negazione di isInvertible
A.isPositiveDefinite()     // tenta Cholesky — successo → PD
A.isPositiveSemiDefinite() // simmetrica + diagonale ≥ 0
A.isDiagonallyDominant()   // |A[i,i]| ≥ Σ_{j≠i} |A[i,j]|
A.hasZeroTrace(tol?)       // traccia ≈ 0
A.hasFiniteValues()        // nessun NaN o Infinity
A.isStochastic(tol?)       // colonne sommano a 1, tutti ≥ 0
```

---

## 🧩 Decomposizioni (`Matrix.decomp`)

### LU — Eliminazione Gaussiana senza Pivot

```typescript
const { L, U } = Matrix.decomp.lu(A);
```

- `L` triangolare inferiore con diagonale di 1
- `U` triangolare superiore
- Lancia eccezione se il pivot è < 1e-12 (usa `lup` in quel caso)

### LUP — LU con Pivoting Parziale ✅ (raccomandato)

```typescript
const { L, U, P, swaps } = Matrix.decomp.lup(A);
// P: array di permutazione degli indici riga
// swaps: numero di scambi (per il segno del determinante)
```

- Trova il pivot massimo in modulo ad ogni passo
- Lancia eccezione se la matrice è singolare

### LU con Pivot Totale

```typescript
const { P, L, U } = Matrix.decomp.luPivoting(A);
// P: Matrix (matrice di permutazione esplicita)
```

### Cholesky

```typescript
const { L } = Matrix.decomp.cholesky(A);
// A = L * L^T
// Richiede: simmetrica e definita positiva
```

### QR — Gram-Schmidt Classico

```typescript
const { Q, R } = Matrix.decomp.qr(A);
// A = Q * R
// Q: matrice con colonne ortonormali (m×n)
// R: triangolare superiore (n×n)
```

### Scomposizione Additiva D+L+U

```typescript
const { D, L, U } = Matrix.decomp.decomposeDLU(A);
// A = D + L + U
// D: diagonale, L: triangolare inf. stretta, U: triangolare sup. stretta
// Usata internamente da Jacobi e Gauss-Seidel
```

### Estrazione Triangolare

```typescript
Matrix.tril(A)         // Triangolare inferiore
Matrix.triu(A, k?)     // Triangolare superiore (k = offset diagonale)
```

---

## ⚡ Solver — Sistemi Lineari `Ax = b`

```typescript
const x = A.solve(b, method);
```

### Metodi Disponibili

| Metodo | Tipo | Prerequisiti | Note |
|--------|------|--------------|------|
| `'LUP'` | Diretto | Quadrata, invertibile | **Default**, più robusto |
| `'LU'` | Diretto | Quadrata, no pivot nullo | Più veloce, meno stabile |
| `'CHOLESKY'` | Diretto | Simmetrica, definita positiva | Il più efficiente per SPD |
| `'QR'` | Diretto | Qualsiasi (anche rettangolare) | Robusto ma lento |
| `'JACOBI'` | Iterativo | Diag. dominante o PD | Converge lentamente |
| `'GAUSS-SEIDEL'` | Iterativo | Diag. dominante o PD | Più veloce di Jacobi |

### Dettaglio Algoritmi

**LUP** (default):
```
1. A = P^T L U  (lup)
2. b_perm = P b  (permutazione del vettore b)
3. y = L^{-1} b_perm  (sostituzione in avanti)
4. x = U^{-1} y       (sostituzione indietro)
```

**CHOLESKY**:
```
1. A = L L^T  (cholesky)
2. y = L^{-1} b
3. x = (L^T)^{-1} y
```

**QR**:
```
1. A = Q R  (qr)
2. x = R^{-1} (Q^T b)   (Q ortogonale → Q^{-1} = Q^T)
```

**JACOBI** (formula matriciale):
```
T = -D^{-1}(L+U),  C = D^{-1}b
x_{k+1} = T x_k + C
Arresto: ||x_{k+1} - x_k||_∞ < tol
```

**GAUSS-SEIDEL** (formula matriciale):
```
T = -(D+L)^{-1} U,  C = (D+L)^{-1} b
x_{k+1} = T x_k + C
```

### Parametri Iterativi

```typescript
// Jacobi e Gauss-Seidel accettano tolleranza e max iterazioni:
Matrix.solver.solveJacobiMat(A, b, tol = 1e-10, maxIter = 1000)
Matrix.solver.solveGaussSeidelMat(A, b, tol = 1e-10, maxIter = 1000)
```

---

## 📈 Performance Benchmark

Risultati misurati con `Matrix.benchmark.ts` su matrici quadrate random, media di 3 esecuzioni.

### Tabella Completa (tempi in ms)

| Operazione | 50×50 | 100×100 | 150×150 | 200×200 | 250×250 | 300×300 |
|-----------|------:|--------:|--------:|--------:|--------:|--------:|
| **add** | 0.104 | 0.197 | 0.072 | 0.130 | 0.166 | 0.249 |
| **sub** | 0.085 | 0.020 | 0.046 | 0.100 | 0.105 | 0.155 |
| **mul** | 1.288 | 0.535 | 1.798 | 5.700 | 11.094 | 13.753 |
| **pow** | 0.362 | 1.656 | 5.522 | 12.000 | 24.866 | 41.696 |
| **dotMul** | 0.108 | 0.215 | 0.584 | 0.500 | 0.718 | 1.069 |
| **dotDiv** | 0.093 | 0.286 | 0.398 | 0.450 | 0.433 | 0.709 |
| **dotPow** | 0.104 | 0.193 | 0.086 | 0.200 | 0.178 | 0.283 |
| **trace** | 0.017 | 0.002 | 0.002 | 0.003 | 0.003 | 0.007 |
| **totalSum** | 0.064 | 0.005 | 0.002 | 0.024 | 0.024 | 0.066 |
| **inv** | 2.803 | 1.960 | — | — | 14.996 | 26.586 |
| **det** | 1.354 | 0.361 | — | — | 3.660 | 5.480 |
| **sum** | 0.109 | 0.148 | 0.030 | 0.030 | 0.033 | 0.046 |
| **mean** | 0.047 | 0.200 | 0.036 | 0.038 | 0.036 | 0.096 |
| **max** | 0.098 | 0.067 | 0.069 | 0.400 | 0.790 | 0.574 |
| **min** | 0.086 | 0.059 | 0.060 | 0.370 | 0.365 | 0.518 |
| **t** | 0.070 | 0.130 | 0.120 | 0.120 | 0.120 | 0.186 |
| **reshape** | 0.014 | 0.004 | 0.002 | 0.001 | 0.001 | 0.002 |
| **repmat** | 0.052 | 0.098 | 0.189 | 0.210 | 0.391 | 0.446 |
| **slice** | 0.027 | 0.030 | 0.030 | 0.030 | 0.029 | 0.052 |
| **flip** | 0.031 | 0.014 | 0.048 | 0.090 | 0.101 | 0.128 |
| **rot90** | 0.094 | 0.223 | 0.117 | 0.120 | 0.120 | 0.286 |
| **abs** | 0.067 | 0.197 | 0.320 | 0.380 | 0.393 | 0.686 |
| **sqrt** | 0.063 | 0.115 | 0.260 | 0.350 | 0.361 | 0.535 |
| **round** | 0.087 | 0.201 | 0.310 | 0.380 | 0.377 | 0.469 |
| **isSquare** | 0.003 | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 |
| **isSymmetric** | 0.013 | 0.002 | 0.004 | 0.004 | 0.001 | 0.001 |
| **isUpperTriangular** | 0.002 | 0.001 | 0.002 | 0.002 | 0.000 | 0.001 |
| **isLowerTriangular** | 0.004 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| **isDiagonal** | 0.004 | 0.017 | 0.002 | 0.002 | 0.001 | 0.001 |
| **isIdentity** | 0.009 | 0.001 | 0.001 | 0.002 | 0.001 | 0.001 |
| **isOrthogonal** | 0.003 | 0.005 | 0.005 | 0.005 | 0.005 | 0.006 |
| **isZeroMatrix** | 0.013 | 0.002 | 0.006 | 0.001 | 0.001 | 0.001 |

### Osservazioni Chiave

- **`reshape` è costante (O(1))** — zero-copy, condivide il buffer originale
- **`trace`, `totalSum`** sono < 0.07ms anche a 300×300 — ottimizzati con stride diretto
- **`mul`** scala cubicamente come atteso (O(n³)), ma il cache-ordering `i-k-j` mantiene alta la densità di hit L1
- **`pow(n²)`** scala come `O(n³ log exp)` — a 300×300 con exp=2 impiega ~42ms (2 moltiplicazioni matriciali)
- **Predicati** (`isSquare`, `isSymmetric`, …) sono praticamente gratuiti: < 0.02ms sempre
- **`inv` e `det`** a 300×300 impiegano ~26ms e ~5.5ms rispettivamente (LUP + sostituzione)

### Scaling della Moltiplicazione

```
 50×50  →   1.3 ms
100×100 →   0.5 ms  (JIT warmup avvantaggiato)
150×150 →   1.8 ms
200×200 →   5.7 ms
250×250 →  11.1 ms
300×300 →  13.8 ms
```

---

## 🧪 Benchmark Comparativo dei Solver

Il file `confronti.benchmark.ts` testa tutti i metodi di risoluzione su 6 tipi di matrici e misura tempo, memoria e residuo relativo `||Ax-b||/(||A|| ||x|| + ||b||)`.

### Tipi di Matrici Testati

| Tipo | Costruzione | Metodi applicabili |
|------|-------------|-------------------|
| `SPD` | `M^T M + εI` | CHOLESKY, LU, LUP, QR, JACOBI, GAUSS-SEIDEL |
| `DiagonalDominant` | Random + dominio forzato | JACOBI, GAUSS-SEIDEL, LUP |
| `General` | Random + `0.5·I` | LU, LUP, QR |
| `Wilkinson` | `Matrix.gallery.wilkinson(n)` | LUP, QR |
| `Hanowa` | `Matrix.gallery.hanowa(n)` | LUP, QR |
| `Dorr` | `Matrix.gallery.dorr(n)` | LUP, JACOBI |

### Criteri di Validazione

```
✅ OK         residuo relativo < 1e-5
⚠️ Instabile  residuo relativo ≥ 1e-5
❌ Fallito    risultato NaN
🚫 Non App.  prerequisiti non soddisfatti (es. Cholesky su non-SPD)
```

**Linee guida empiriche:**
- Per matrici **SPD**: preferire CHOLESKY (più veloce, stabile)
- Per sistemi **diagonalmente dominanti**: GAUSS-SEIDEL converge più rapidamente di JACOBI
- Per matrici **generali ben condizionate**: LUP è la scelta sicura
- Evitare LU (senza pivot) su matrici di Wilkinson o mal condizionate

---

## 💡 Tecniche di Ottimizzazione Implementate


### 1. Ordine i-k-j per la Moltiplicazione
L'ordine naturale `i-j-k` causa cache miss su `B` a ogni cambio di `j`. L'ordine `i-k-j` carica `A[i,k]` una volta e scorre `B[k,:]` sequenzialmente:
```typescript
for (let k = 0; k < K; k++) {
    const aik = A[iOff + k];   // caricato una volta nel registro
    for (let j = 0; j < N; j++) C[outOff+j] += aik * B[kOff+j];  // accesso lineare
}
```

### 2. Float64Array Contiguo
Tutti i dati sono in un singolo buffer piatto, row-major. Nessun array di array, nessun boxing, prefetch automatico della CPU.

### 3. Broadcasting Senza Copia
`add`, `sub`, `dotMul` etc. evitano di materializzare un buffer espanso: il valore del vettore riga/colonna viene letto e applicato inline.

### 4. Reshape O(1)
```typescript
return new Matrix(r, c, this.data);  // stesso Float64Array, nuovi metadati
```

### 5. Esponenziazione Rapida
`pow(k)` usa binary exponentiation: O(log k) moltiplicazioni invece di O(k).

---

## 🛠️ Esempi di Utilizzo

```typescript
import { Matrix } from './src';

// Creazione
const A = Matrix.random(4, 4, { type: 'uniform', min: 1, max: 10 });
const I = Matrix.identity(4);

// Operazioni
const B = A.add(I.mul(0.5));
const C = A.mul(B);
const d = A.det();
const tr = A.trace();

// Decomposizione
const { L, U, P } = Matrix.decomp.lup(A);

// Soluzione sistema lineare
const b = Matrix.random(4, 1);
const x = A.solve(b, 'LUP');

// Verifica residuo
const residual = A.mul(x).sub(b).norm('inf');
console.log('Residuo:', residual);  // dovrebbe essere ~1e-14

// Galleria
const H = Matrix.gallery.hilbert(5);
console.log('Hilbert è SPD?', H.isPositiveDefinite());  // true

// Statistiche
const { value, index } = A.max(1);  // massimo per colonna
const rowSums = A.sum(2);           // somma per riga

// Trasformazioni
const AT = A.t();
const A2 = A.reshape(2, 8);        // O(1), nessuna copia
const sub = A.slice(0, 2, 0, 2);   // sottomatrice 2×2
```

---

## 📋 API Reference Rapida

### Metodi dell'istanza (`new Matrix` / `Matrix.random(...)`)

| Categoria | Metodo | Ritorna |
|-----------|--------|---------|
| **Aritmetica** | `add(B)`, `sub(B)`, `mul(B)`, `pow(k)` | `Matrix` |
| **Dot** | `dotMul(B)`, `dotDiv(B)`, `dotPow(e)` | `Matrix` |
| **Algebra** | `det()`, `trace()`, `inv()`, `norm(t)` | `number` / `Matrix` |
| **Statistiche** | `sum(dim)`, `mean(dim)` | `Matrix` |
| | `max(dim)`, `min(dim)` | `{value, index}` |
| | `totalSum()` | `number` |
| **Transform** | `t()`, `reshape(r,c)`, `slice(...)` | `Matrix` |
| | `repmat(r,c)`, `flip(d)`, `rot90(k)` | `Matrix` |
| **Unary** | `abs()`, `sqrt()`, `round()` | `Matrix` |
| **Predicati** | `isSquare()`, `isSymmetric()`, ... | `boolean` |
| | `isPositiveDefinite()`, `isInvertible()`, ... | `boolean` |
| **Solver** | `solve(b, method)` | `Matrix` |
| **Base** | `get(i,j)`, `set(i,j,v)`, `clone()` | — |
| | `toString()`, `equals(B, tol)` | `string` / `boolean` |

### Factory Statiche

```typescript
Matrix.zeros(r,c)          Matrix.ones(r,c)
Matrix.identity(n)         Matrix.diag(n,k)
Matrix.diagFromArray(arr)  Matrix.random(n,m,opts)
Matrix.sparse(n,m,opts)    Matrix.toeplitz(c,r?)
Matrix.hankel(c,r?)        Matrix.vander(v)
Matrix.tril(A)             Matrix.triu(A,k?)
```

### Namespace Statici

```typescript
Matrix.gallery.*    // 24 matrici note
Matrix.decomp.*     // lu, lup, luPivoting, cholesky, qr, decomposeDLU, tril, triu
Matrix.solver.*     // solve, solveLowerTriangular, solveUpperTriangular,
                    // solveJacobiMat, solveGaussSeidelMat
```

---

## 📦 Dipendenze

| Pacchetto | Versione | Uso |
|-----------|----------|-----|
| `vitest` | ^4.1.0 | Test runner |
| `plotly.js` | ^3.4.0 | Visualizzazione benchmark |
| `jsdom` | ^29.0.0 | DOM virtuale per test |
| `@types/node` | ^25.5.0 | Tipi TypeScript per Node |

---

## 📝 Note Tecniche

- **Precisione:** `Float64Array` usa IEEE 754 double precision (64-bit), identica al `double` di C/Fortran
- **Indici:** interni 0-based; le funzioni statistiche (`max`, `min`) restituiscono indici 1-based per compatibilità MATLAB
- **Immutabilità:** tutti i metodi creano una **nuova** matrice (eccetto i metodi in-place interni come `divideInPlace`)
- **Singolarità:** `lup` lancia eccezione se `|pivot| < EPS`; `lu` se `|pivot| < 1e-12`
- **Eccezioni:** dimensioni incompatibili → `Error` con messaggio descrittivo

---

*Libreria sviluppata per il corso di Metodi del Calcolo Scientifico.*
