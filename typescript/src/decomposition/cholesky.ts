// decomposition/cholesky.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Decomposizione di Cholesky: A = L * L^T.
 * Richiede A simmetrica e definita positiva.
 */
export function cholesky<T extends INumeric<T>>(A: Matrix<T>): { L: Matrix<T> } {
    if (A.rows !== A.cols) throw new Error("cholesky: matrice non quadrata.");
    if (!A.isSymmetric())  throw new Error("cholesky: matrice non simmetrica.");

    const N = A.rows;
    const L = A.like(N, N);

    for (let j = 0; j < N; j++) {
        // Elemento diagonale: L[j,j] = sqrt( A[j,j] - sum(L[j,k]^2, k<j) )
        let diagSum = A.zero;
        for (let k = 0; k < j; k++) {
            diagSum = diagSum.add(L.get(j, k).multiply(L.get(j, k)));
        }
        const d = A.get(j, j).subtract(diagSum);

        // Verifica definita positiva
        if (d.negate().greaterThan(A.zero)) {
            throw new Error("cholesky: la matrice non è definita positiva.");
        }
        L.set(j, j, d.sqrt());

        // Colonna j, righe sotto la diagonale
        for (let i = j + 1; i < N; i++) {
            let offSum = A.zero;
            for (let k = 0; k < j; k++) {
                offSum = offSum.add(L.get(i, k).multiply(L.get(j, k)));
            }
            L.set(i, j, A.get(i, j).subtract(offSum).divide(L.get(j, j)));
        }
    }

    return { L };
}
