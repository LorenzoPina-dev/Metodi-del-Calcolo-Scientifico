import { Matrix } from "../core";
import { identity, zeros } from "../init";

export function lup(A: Matrix): { L: Matrix; U: Matrix; P: number[] } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("Square only");
    const AClone = A.clone();
    const P = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < n; i++) {
        let max = i, maxVal = Math.abs(AClone.get(i, i));
        for (let k = i + 1; k < n; k++) {
            const val = Math.abs(AClone.get(k, i));
            if (val > maxVal) { maxVal = val; max = k; }
        }
        if (Matrix.isZero(maxVal)) throw new Error("Singular matrix");
        if (max !== i) {
            for (let j = 0; j < n; j++) {
                const tmp = AClone.get(i, j);
                AClone.set(i, j, AClone.get(max, j));
                AClone.set(max, j, tmp);
            }
            [P[i], P[max]] = [P[max], P[i]];
        }
        for (let j = i + 1; j < n; j++) {
            const factor = AClone.get(j, i) / AClone.get(i, i);
            AClone.set(j, i, factor);
            for (let k = i + 1; k < n; k++) AClone.set(j, k, AClone.get(j, k) - factor * AClone.get(i, k));
        }
    }
    const L = identity(n), U = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            if (i > j) L.set(i, j, AClone.get(i, j));
            else U.set(i, j, AClone.get(i, j));
    return { L, U, P };
}