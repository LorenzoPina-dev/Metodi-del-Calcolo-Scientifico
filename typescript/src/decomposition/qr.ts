// decomposition/qr.ts
//
// Gram-Schmidt Modificato (MGS) con prodotto interno hermitiano corretto.
//
// Prodotto interno: <u, v> = Σ conj(u_i) * v_i
//   → per tipi reali (Float64M, Rational): conj(x) = x → uguale al classico
//   → per Complex: usa il vero prodotto interno hermitiano
//
// Questo garantisce che Q^H * Q = I sia valido sia per matrici reali che complesse.
//
import { Matrix } from "..";
import { INumeric } from "../type";

export function qr<T extends INumeric<T>>(A: Matrix<T>): { Q: Matrix<T>; R: Matrix<T> } {
    const m = A.rows, n = A.cols;

    const Q = A.like(m, n);
    const R = A.like(n, n);
    const W = A.clone();  // copia di lavoro modificata in-place da MGS

    const qd = Q.data, rd = R.data, wd = W.data;

    for (let k = 0; k < n; k++) {

        // ---- Norma² hermitiana della colonna k: <W_k, W_k> = Σ conj(w_i)*w_i = Σ |w_i|² ----
        let normSq = A.zero;
        for (let i = 0; i < m; i++) {
            const v = wd[i * n + k];
            // conj(v) * v = |v|²  (reale e positivo per qualsiasi T)
            normSq = normSq.add(v.conjugate().multiply(v));
        }

        const normK = normSq.sqrt();  // ||W_k|| reale (sqrt di valore reale ≥ 0)

        if (A.isZero(normK)) {
            throw new Error(`qr: colonne linearmente dipendenti alla colonna ${k}.`);
        }

        rd[k * n + k] = normK;

        // ---- q_k = W_k / ||W_k|| ----
        for (let i = 0; i < m; i++) {
            qd[i * n + k] = wd[i * n + k].divide(normK);
        }

        // ---- MGS: aggiorna le colonne successive di W ----
        // r_{kj} = <q_k, W_j> = Σ conj(q_k_i) * W_j_i
        // W_j  -= r_{kj} * q_k
        for (let j = k + 1; j < n; j++) {
            let dot = A.zero;
            for (let i = 0; i < m; i++) {
                // conj(q_k_i) * w_j_i  →  prodotto interno hermitiano
                dot = dot.add(qd[i * n + k].conjugate().multiply(wd[i * n + j]));
            }
            rd[k * n + j] = dot;
            for (let i = 0; i < m; i++) {
                wd[i * n + j] = wd[i * n + j].subtract(dot.multiply(qd[i * n + k]));
            }
        }
    }

    return { Q, R };
}
