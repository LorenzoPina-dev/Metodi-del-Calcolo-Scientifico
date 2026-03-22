// solver/ldlt.ts
//
// Fattorizzazione LDL^T — per matrici simmetriche (non necessariamente SPD).
//
// A = L * D * L^T
// dove L è triangolare inferiore unitaria e D è diagonale.
//
// Più generale di Cholesky (funziona anche quando A è indefinita purché
// i pivot non siano zero) e più economica di LU (sfrutta la simmetria:
// esegue circa n³/6 operazioni vs n³/3 di LU).
//
// Algoritmo (fattorizzazione di Bunch-Parlett senza pivoting):
//   Per j = 0..n-1:
//     D[j,j] = A[j,j] - Σ_{k<j} L[j,k]² * D[k,k]
//     Per i = j+1..n-1:
//       L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]*L[j,k]*D[k,k]) / D[j,j]
//
import { Matrix } from "..";
import { INumeric } from "../type";

export function ldlt<T extends INumeric<T>>(A: Matrix<T>): { L: Matrix<T>; D: Matrix<T> } {
    if (A.rows !== A.cols) throw new Error("ldlt: matrice non quadrata.");
    if (!A.isSymmetric())  throw new Error("ldlt: matrice non simmetrica.");

    const n = A.rows;
    const L = A.likeIdentity(n);      // triangolare inferiore unitaria
    const D = A.like(n, n);           // diagonale
    const ad = A.data, ld = L.data, dd = D.data;

    for (let j = 0; j < n; j++) {
        // D[j,j] = A[j,j] - Σ_{k<j} L[j,k]² * D[k,k]
        let djj = ad[j * n + j];
        for (let k = 0; k < j; k++) {
            const ljk = ld[j * n + k];
            djj = djj.subtract(ljk.multiply(ljk).multiply(dd[k * n + k]));
        }
        if (A.isZero(djj)) throw new Error(`ldlt: pivot nullo alla colonna ${j}. Matrice singolare o quasi.`);
        dd[j * n + j] = djj;

        // L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]*L[j,k]*D[k,k]) / D[j,j]
        for (let i = j + 1; i < n; i++) {
            let lij = ad[i * n + j];
            for (let k = 0; k < j; k++) {
                lij = lij.subtract(ld[i * n + k].multiply(ld[j * n + k]).multiply(dd[k * n + k]));
            }
            ld[i * n + j] = lij.divide(djj);
        }
    }

    return { L, D };
}

/**
 * Risolve A*x = b sfruttando la fattorizzazione LDL^T.
 * Sequenza: L*y = b  →  D*z = y  →  L^T*x = z
 */
export function solveLDLT<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>
): Matrix<T> {
    const { L, D } = ldlt(A);
    const n = A.rows;
    const ld = L.data, dd = D.data, bd = b.data;

    // Passo 1: L * y = b  (sostituzione in avanti)
    const y = b.like(n, 1);
    const yd = y.data;
    yd[0] = bd[0];
    for (let i = 1; i < n; i++) {
        let s = bd[i];
        for (let k = 0; k < i; k++) s = s.subtract(ld[i * n + k].multiply(yd[k]));
        yd[i] = s;                    // diagonale di L = 1 → nessuna divisione
    }

    // Passo 2: D * z = y  (D è diagonale)
    const z = b.like(n, 1);
    const zd = z.data;
    for (let i = 0; i < n; i++) zd[i] = yd[i].divide(dd[i * n + i]);

    // Passo 3: L^T * x = z  (sostituzione all'indietro con L^T)
    const x = b.like(n, 1);
    const xd = x.data;
    xd[n - 1] = zd[n - 1];
    for (let i = n - 2; i >= 0; i--) {
        let s = zd[i];
        // L^T[i, k] = L[k, i]  per k > i
        for (let k = i + 1; k < n; k++) s = s.subtract(ld[k * n + i].multiply(xd[k]));
        xd[i] = s;                    // diagonale di L^T = 1
    }

    return x;
}
