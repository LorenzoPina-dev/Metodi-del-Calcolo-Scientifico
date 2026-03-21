import { Matrix } from "..";
import { identity } from "../init";
import { INumeric } from "../type";

export function cholesky(A: Matrix): { L: Matrix } {
    if (A.rows !== A.cols) throw new Error("Matrix must be square");
    if (!A.isSymmetric()) throw new Error("Matrix is not symmetric");
    const N = A.rows;
    const L = identity(N);
    for (let n = 0; n < N; n++) {
        let sum2 : INumeric = INumeric.zero;
        for (let m = 0; m <= n; m++) {
            if (n === m) {
                let s = A.get(n, n).subtract(sum2);
                if (s.lessThan(INumeric.zero)) throw new Error("Matrix is not positive definite");
                L.set(n, n, s.sqrt());
            } else {
                let sum : INumeric = INumeric.zero;
                for (let k = 0; k < m; k++) sum = sum.add(L.get(n, k).multiply(L.get(m, k)));
                L.set(n, m, A.get(n, m).subtract(sum).divide(L.get(m, m)));
                sum2 = sum2.add(L.get(n, m).multiply(L.get(n, m)));
            }
        }
    }
    return { L };
}