import { Matrix } from "..";
import { identity, zeros } from "../init";

export function lup(A: Matrix): { L: Matrix; U: Matrix; P: number[], swaps: number } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("Square only");
    const AClone = A.clone();
    const P = Array.from({ length: n }, (_, i) => i);
    let swaps = 0;

    for (let i = 0; i < n; i++) {
        let max = i, maxVal = AClone.get(i, i).abs();
        for (let k = i + 1; k < n; k++) {
            const val = AClone.get(k, i).abs();
            if (val.greaterThan(maxVal)) { maxVal = val; max = k; }
        }
        if (Matrix.isZero(maxVal)) throw new Error("Singular matrix");
        if (max !== i) {
            swaps++;
            for (let j = 0; j < n; j++) {
                const tmp = AClone.get(i, j);
                AClone.set(i, j, AClone.get(max, j));
                AClone.set(max, j, tmp);
            }
            [P[i], P[max]] = [P[max], P[i]];
        }
        for (let j = i + 1; j < n; j++) {
            const factor = AClone.get(j, i).divide(AClone.get(i, i));
            AClone.set(j, i, factor);
            for (let k = i + 1; k < n; k++) AClone.set(j, k, AClone.get(j, k).subtract(factor.multiply(AClone.get(i, k))));
        }
    }
    const L = identity(n), U = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            if (i > j) L.set(i, j, AClone.get(i, j));
            else U.set(i, j, AClone.get(i, j));
    return { L, U, P, swaps };
}