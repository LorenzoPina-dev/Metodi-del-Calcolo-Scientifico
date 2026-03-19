import { Matrix } from "..";
import { identity } from "../init";

export function cholesky(A: Matrix): { L: Matrix } {
    if (A.rows !== A.cols) throw new Error("Matrix must be square");
    if (!A.isSymmetric()) throw new Error("Matrix is not symmetric");
    const N = A.rows;
    const L = identity(N);
    for (let n = 0; n < N; n++) {
        let sum2 = 0;
        for (let m = 0; m <= n; m++) {
            if (n === m) {
                let s = A.get(n, n) - sum2;
                if (s < 0) throw new Error("Matrix is not positive definite");
                L.set(n, n, Math.sqrt(s));
            } else {
                let sum = 0;
                for (let k = 0; k < m; k++) sum += L.get(n, k) * L.get(m, k);
                L.set(n, m, (A.get(n, m) - sum) / L.get(m, m));
                sum2 += L.get(n, m) * L.get(n, m);
            }
        }
    }
    return { L };
}