// solver/gausSeidelMat.ts
//
// Forma element-wise — aggiorna x[i] in-place ad ogni passo usando i valori
// già aggiornati nella stessa iterazione (questo È la differenza con Jacobi).
// O(n²) per iterazione, nessuna costruzione di T o C.
//
import { Matrix } from "..";
import { INumeric } from "../type";
import { _hasConverged } from "./_hasConverged";

export function solveGaussSeidelMat<T extends INumeric<T>>(
    A: Matrix<T>,
    b: Matrix<T>,
    tol: number = 1e-10,
    maxIter: number = 1000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveGaussSeidelMat: matrice non quadrata.");

    const ad = A.data;
    const bd = b.data;
    let x = A.like(n, 1);

    for (let iter = 0; iter < maxIter; iter++) {
        const xPrev = x.clone();
        const xd = x.data;

        for (let i = 0; i < n; i++) {
            const off = i * n;
            let s = bd[i];
            // j < i → usa xd[j] già aggiornato in questa iterazione
            for (let j = 0; j < i; j++)  s = s.subtract(ad[off + j].multiply(xd[j]));
            // j > i → usa xd[j] dell'iterazione precedente
            for (let j = i + 1; j < n; j++) s = s.subtract(ad[off + j].multiply(xd[j]));
            xd[i] = s.divide(ad[off + i]);
        }

        if (_hasConverged(xPrev, x, tol)) return x;
    }
    return x;
}
