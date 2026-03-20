import { Matrix } from "..";
import { identity } from "../init";

export function lu(A: Matrix): { L: Matrix; U: Matrix } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("Matrix must be square");

    const U = A.clone();
    const L = Matrix.identity(n);

    const EPS = 1e-12;

    for (let k = 0; k < n; k++) {
        const pivot = U.get(k, k);

        // Controllo pivot
        if (Math.abs(pivot) < EPS) {
            throw new Error("Zero or near-zero pivot encountered. Use LUP instead.");
        }

        for (let i = k + 1; i < n; i++) {
            const factor = U.get(i, k) / pivot;

            // Salva il moltiplicatore in L
            L.set(i, k, factor);

            // Aggiorna riga di U
            for (let j = k; j < n; j++) {
                const value = U.get(i, j) - factor * U.get(k, j);
                U.set(i, j, value);
            }
        }
    }

    return { L, U };
}