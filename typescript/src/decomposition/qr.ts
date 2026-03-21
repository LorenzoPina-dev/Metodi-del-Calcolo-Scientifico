// decomposition/qr.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Decomposizione QR tramite Gram-Schmidt classico.
 * A = Q * R, con Q ortogonale (m×n) e R triangolare superiore (n×n).
 */
export function qr<T extends INumeric<T>>(A: Matrix<T>): { Q: Matrix<T>; R: Matrix<T> } {
    const m = A.rows;
    const n = A.cols;

    const Q = A.like(m, n);
    const R = A.like(n, n);
    const W = A.clone();   // copia di lavoro (verrà modificata da Gram-Schmidt)

    for (let k = 0; k < n; k++) {
        // --- Norma della colonna k ---
        let normSq = A.zero;
        for (let i = 0; i < m; i++) {
            const v = W.get(i, k);
            normSq = normSq.add(v.multiply(v));
        }
        const normK = normSq.sqrt();

        if (A.isZero(normK)) {
            throw new Error("qr: colonne linearmente dipendenti alla colonna " + k + ".");
        }

        R.set(k, k, normK);

        // q_k = a_k / ||a_k||
        for (let i = 0; i < m; i++) {
            Q.set(i, k, W.get(i, k).divide(normK));
        }

        // Proiezioni sulle colonne successive: a_j -= (q_k · a_j) * q_k
        for (let j = k + 1; j < n; j++) {
            let dot = A.zero;
            for (let i = 0; i < m; i++) {
                dot = dot.add(Q.get(i, k).multiply(W.get(i, j)));
            }
            R.set(k, j, dot);
            for (let i = 0; i < m; i++) {
                W.set(i, j, W.get(i, j).subtract(dot.multiply(Q.get(i, k))));
            }
        }
    }

    return { Q, R };
}
