import { Matrix } from "../core";
import { identity, zeros } from "../init";

export function qr(A: Matrix): { Q: Matrix; R: Matrix } {
    const m = A.rows;
    const n = A.cols;
    const Q = zeros(m, n);
    const R = zeros(n, n);
    const AClone = A.clone();

    for (let k = 0; k < n; k++) {
        // r_kk = ||a_k||
        let norm = 0;
        for (let i = 0; i < m; i++) {
            const val = AClone.get(i, k);
            norm += val * val;
        }
        norm = Math.sqrt(norm);
        if (norm === 0) throw new Error("Matrix has linearly dependent columns");
        R.set(k, k, norm);

        // q_k = a_k / r_kk
        for (let i = 0; i < m; i++) {
            Q.set(i, k, AClone.get(i, k) / norm);
        }

        // Gram-Schmidt: aggiornamento colonne successive
        for (let j = k + 1; j < n; j++) {
            let r_kj = 0;
            for (let i = 0; i < m; i++) {
                r_kj += Q.get(i, k) * AClone.get(i, j);
            }
            R.set(k, j, r_kj);
            for (let i = 0; i < m; i++) {
                AClone.set(i, j, AClone.get(i, j) - r_kj * Q.get(i, k));
            }
        }
    }

    return { Q, R };
}