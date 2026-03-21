// decomposition/lup.ts
//
// Ottimizzazioni:
//  1. Ricerca del pivot con accesso diretto a data[].
//  2. Scambio righe: blocco intero copiato con splice/slice anziché elemento per elemento.
//  3. Eliminazione: accesso lineare agli array.
//
import { Matrix } from "..";
import { INumeric } from "../type";

export function lup<T extends INumeric<T>>(
    A: Matrix<T>
): { L: Matrix<T>; U: Matrix<T>; P: number[]; swaps: number } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("lup: matrice non quadrata.");

    const W = A.clone();
    const wd = W.data;
    const P = Array.from({ length: n }, (_, i) => i);
    let swaps = 0;

    for (let i = 0; i < n; i++) {
        // ---- Ricerca pivot massimo (accesso lineare alla colonna) ----
        let maxVal = wd[i * n + i].abs();
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            const v = wd[k * n + i].abs();
            if (v.greaterThan(maxVal)) { maxVal = v; maxRow = k; }
        }
        if (W.isZero(maxVal)) throw new Error("lup: matrice singolare.");

        // ---- Scambio righe in blocco ----
        if (maxRow !== i) {
            swaps++;
            const iOff = i * n, mOff = maxRow * n;
            for (let j = 0; j < n; j++) {
                const t = wd[iOff + j]; wd[iOff + j] = wd[mOff + j]; wd[mOff + j] = t;
            }
            [P[i], P[maxRow]] = [P[maxRow], P[i]];
        }

        // ---- Eliminazione Gaussiana ----
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

    // ---- Estrazione L e U ----
    const L = A.likeIdentity(n);
    const U = A.like(n, n);
    const ld = L.data, ud = U.data;

    for (let i = 0; i < n; i++) {
        const off = i * n;
        for (let j = 0; j < n; j++) {
            if (i > j)      ld[off + j] = wd[off + j];
            else            ud[off + j] = wd[off + j];
        }
    }

    return { L, U, P, swaps };
}
