// decomposition/lu_total.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function lu_total<T extends INumeric<T>>(
    A: Matrix<T>
): { L: Matrix<T>; U: Matrix<T>; PR: number[]; PC: number[]; swaps: number } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("lu_total: matrice non quadrata.");

    const W = A.clone();
    const wd = W.data;
    const PR = Array.from({ length: n }, (_, i) => i);
    const PC = Array.from({ length: n }, (_, i) => i);
    let swaps = 0;

    for (let i = 0; i < n; i++) {
        // Pivot massimo nel sotto-blocco [i..n)×[i..n)
        let maxVal = W.isZero(wd[i * n + i]) ? W.zero.abs() : wd[i * n + i].abs();
        let maxR = i, maxC = i;
        for (let r = i; r < n; r++) {
            for (let c = i; c < n; c++) {
                const v = wd[r * n + c].abs();
                if (v.greaterThan(maxVal)) { maxVal = v; maxR = r; maxC = c; }
            }
        }
        if (W.isZero(maxVal)) throw new Error("lu_total: matrice singolare.");

        // Scambio righe
        if (maxR !== i) {
            swaps++;
            for (let j = 0; j < n; j++) {
                const t = wd[i * n + j]; wd[i * n + j] = wd[maxR * n + j]; wd[maxR * n + j] = t;
            }
            [PR[i], PR[maxR]] = [PR[maxR], PR[i]];
        }
        // Scambio colonne
        if (maxC !== i) {
            swaps++;
            for (let j = 0; j < n; j++) {
                const t = wd[j * n + i]; wd[j * n + i] = wd[j * n + maxC]; wd[j * n + maxC] = t;
            }
            [PC[i], PC[maxC]] = [PC[maxC], PC[i]];
        }

        // Eliminazione
        const pivot = wd[i * n + i];
        for (let j = i + 1; j < n; j++) {
            const jOff = j * n;
            const factor = wd[jOff + i].divide(pivot);
            wd[jOff + i] = factor;
            for (let k = i + 1; k < n; k++) {
                wd[jOff + k] = wd[jOff + k].subtract(factor.multiply(wd[i * n + k]));
            }
        }
    }

    const L = A.likeIdentity(n);
    const U = A.like(n, n);
    const ld = L.data, ud = U.data;
    for (let i = 0; i < n; i++) {
        const off = i * n;
        for (let j = 0; j < n; j++) {
            if (i > j) ld[off + j] = wd[off + j];
            else       ud[off + j] = wd[off + j];
        }
    }
    return { L, U, PR, PC, swaps };
}
