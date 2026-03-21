// decomposition/lu.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function lu<T extends INumeric<T>>(A: Matrix<T>): { L: Matrix<T>; U: Matrix<T> } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("lu: matrice non quadrata.");

    const U = A.clone();
    const L = A.likeIdentity(n);
    const ud = U.data, ld = L.data;
    const EPS = 1e-12;

    for (let k = 0; k < n; k++) {
        const pivot = ud[k * n + k];
        if (pivot.isNearZero(EPS)) {
            throw new Error(`lu: pivot nullo alla colonna ${k}. Usare lup().`);
        }
        for (let i = k + 1; i < n; i++) {
            const iOff = i * n;
            const factor = ud[iOff + k].divide(pivot);
            ld[iOff + k] = factor;
            for (let j = k; j < n; j++) {
                ud[iOff + j] = ud[iOff + j].subtract(factor.multiply(ud[k * n + j]));
            }
        }
    }
    return { L, U };
}
