// decomposition/cholesky.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Decomposizione di Cholesky: A = L * L^T  (solo triangolare inferiore).
 * Accesso diretto a data[] — elimina il sovraccarico di get/set per matrici grandi.
 */
export function cholesky<T extends INumeric<T>>(A: Matrix<T>): { L: Matrix<T> } {
    if (A.rows !== A.cols) throw new Error("cholesky: matrice non quadrata.");
    if (!A.isSymmetric())  throw new Error("cholesky: matrice non simmetrica.");

    const N = A.rows;
    const L = A.like(N, N);
    const ad = A.data, ld = L.data;

    for (let j = 0; j < N; j++) {
        // L[j,j] = sqrt( A[j,j] - Σ L[j,k]² )
        let diagSum = A.zero;
        for (let k = 0; k < j; k++) {
            const ljk = ld[j * N + k];
            diagSum = diagSum.add(ljk.multiply(ljk));
        }
        const d = ad[j * N + j].subtract(diagSum);
        if (d.negate().greaterThan(A.zero)) {
            throw new Error("cholesky: la matrice non è definita positiva.");
        }
        const ljj = d.sqrt();
        ld[j * N + j] = ljj;

        // Colonna j, righe i > j
        for (let i = j + 1; i < N; i++) {
            let offSum = A.zero;
            for (let k = 0; k < j; k++) {
                offSum = offSum.add(ld[i * N + k].multiply(ld[j * N + k]));
            }
            ld[i * N + j] = ad[i * N + j].subtract(offSum).divide(ljj);
        }
    }

    return { L };
}
