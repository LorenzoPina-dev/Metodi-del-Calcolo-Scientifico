// decomposition/lup.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Decomposizione LUP con pivoting parziale.
 * Restituisce L, U (entrambe dello stesso tipo T di A) e il vettore di permutazione P.
 */
export function lup<T extends INumeric<T>>(
    A: Matrix<T>
): { L: Matrix<T>; U: Matrix<T>; P: number[]; swaps: number } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("lup: matrice non quadrata.");

    const W = A.clone();          // copia di lavoro
    const P = Array.from({ length: n }, (_, i) => i);
    let swaps = 0;

    for (let i = 0; i < n; i++) {
        // --- Ricerca del pivot massimo in colonna i ---
        let maxVal = W.get(i, i).abs();
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            const v = W.get(k, i).abs();
            if (v.greaterThan(maxVal)) { maxVal = v; maxRow = k; }
        }

        if (W.isZero(maxVal)) throw new Error("lup: matrice singolare.");

        // --- Scambio righe ---
        if (maxRow !== i) {
            swaps++;
            for (let j = 0; j < n; j++) {
                const tmp = W.get(i, j);
                W.set(i, j, W.get(maxRow, j));
                W.set(maxRow, j, tmp);
            }
            [P[i], P[maxRow]] = [P[maxRow], P[i]];
        }

        // --- Eliminazione ---
        for (let j = i + 1; j < n; j++) {
            const factor = W.get(j, i).divide(W.get(i, i));
            W.set(j, i, factor);                      // salva il moltiplicatore in L
            for (let k = i + 1; k < n; k++) {
                W.set(j, k, W.get(j, k).subtract(factor.multiply(W.get(i, k))));
            }
        }
    }

    // --- Estrazione L e U ---
    const L = A.likeIdentity(n);
    const U = A.like(n, n);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i > j)      L.set(i, j, W.get(i, j));
            else            U.set(i, j, W.get(i, j));
        }
    }

    return { L, U, P, swaps };
}
