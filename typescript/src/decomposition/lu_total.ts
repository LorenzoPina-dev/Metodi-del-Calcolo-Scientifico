// decomposition/lu_total.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Decomposizione LU con pivoting totale (riga + colonna).
 * Restituisce L, U e i vettori di permutazione PR (righe) e PC (colonne).
 */
export function lu_total<T extends INumeric<T>>(
    A: Matrix<T>
): { L: Matrix<T>; U: Matrix<T>; PR: number[]; PC: number[]; swaps: number } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("lu_total: matrice non quadrata.");

    const W = A.clone();
    const PR = Array.from({ length: n }, (_, i) => i);
    const PC = Array.from({ length: n }, (_, i) => i);
    let swaps = 0;

    for (let i = 0; i < n; i++) {
        // Ricerca pivot massimo nel sotto-blocco [i..n)×[i..n)
        let maxVal = W.get(i, i).abs();
        let maxR = i, maxC = i;
        for (let r = i; r < n; r++) {
            for (let c = i; c < n; c++) {
                const v = W.get(r, c).abs();
                if (v.greaterThan(maxVal)) { maxVal = v; maxR = r; maxC = c; }
            }
        }

        if (W.isZero(maxVal)) throw new Error("lu_total: matrice singolare.");

        // Scambio righe
        if (maxR !== i) {
            swaps++;
            for (let j = 0; j < n; j++) {
                const t = W.get(i, j); W.set(i, j, W.get(maxR, j)); W.set(maxR, j, t);
            }
            [PR[i], PR[maxR]] = [PR[maxR], PR[i]];
        }
        // Scambio colonne
        if (maxC !== i) {
            swaps++;
            for (let j = 0; j < n; j++) {
                const t = W.get(j, i); W.set(j, i, W.get(j, maxC)); W.set(j, maxC, t);
            }
            [PC[i], PC[maxC]] = [PC[maxC], PC[i]];
        }

        // Eliminazione
        for (let j = i + 1; j < n; j++) {
            const factor = W.get(j, i).divide(W.get(i, i));
            W.set(j, i, factor);
            for (let k = i + 1; k < n; k++) {
                W.set(j, k, W.get(j, k).subtract(factor.multiply(W.get(i, k))));
            }
        }
    }

    const L = A.likeIdentity(n);
    const U = A.like(n, n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i > j) L.set(i, j, W.get(i, j));
            else       U.set(i, j, W.get(i, j));
        }
    }

    return { L, U, PR, PC, swaps };
}
