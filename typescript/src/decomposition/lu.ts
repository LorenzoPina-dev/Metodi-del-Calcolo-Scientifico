import { Matrix } from "../core";
import { identity } from "../init";

export function lu(A: Matrix): { L: Matrix; U: Matrix } {
    const N = A.rows;
    if (N !== A.cols) throw new Error("Matrix must be square");
    let A_old = A.clone(), L = identity(N);
    for (let n = 0; n < N - 1; n++) {
        const Mn = identity(N);
        const Mn_inv = identity(N);
        for (let i = n + 1; i < N; i++) {
            const val = A_old.get(i, n) / A_old.get(n, n);
            Mn.set(i, n, -val);
            Mn_inv.set(i, n, val);
        }
        A_old = Mn.multiply(A_old);
        L = L.multiply(Mn_inv);
    }
    return { L, U: A_old };
}