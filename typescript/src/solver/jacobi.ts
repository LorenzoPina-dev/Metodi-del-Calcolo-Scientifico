// solver/jacobi.ts
//
// Forma element-wise — O(n²) per iterazione anziché O(n³) nella forma matriciale.
// Nessuna costruzione di T o C — zero allocazioni matriciali in setup.
//
import { Matrix } from "..";
import { INumeric } from "../type";
import { _hasConverged } from "./_hasConverged";

export function solveJacobiMat<T extends INumeric<T>>(
    A: Matrix<T>,
    b: Matrix<T>,
    tol: number = 1e-10,
    maxIter: number = 1000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveJacobiMat: matrice non quadrata.");

    const ad = A.data;
    const bd = b.data;
    let x = A.like(n, 1);
    const xd = x.data;

    for (let iter = 0; iter < maxIter; iter++) {
        const xNext = A.like(n, 1);
        const nd = xNext.data;

        for (let i = 0; i < n; i++) {
            const off = i * n;
            let s = bd[i];
            for (let j = 0; j < n; j++) {
                if (j !== i) s = s.subtract(ad[off + j].multiply(xd[j]));
            }
            nd[i] = s.divide(ad[off + i]);
        }

        if (_hasConverged(x, xNext, tol)) return xNext;
        x = xNext;
    }
    return x;
}
