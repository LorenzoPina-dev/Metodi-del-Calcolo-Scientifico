// solver/jacobi.ts
import { Matrix } from "..";
import { INumeric } from "../type";
import { decomposeDLU } from "../decomposition/dlu";
import { inverseDiagonal } from "../algoritm/inverse";
import { _hasConverged } from "./_hasConverged";

/**
 * Risolve A*x = b con il metodo di Jacobi (forma matriciale).
 * Formula: x_{k+1} = D^{-1} * (b - (L+U) * x_k)
 */
export function solveJacobiMat<T extends INumeric<T>>(
    A: Matrix<T>,
    b: Matrix<T>,
    tol: number = 1e-10,
    maxIter: number = 1000
): Matrix<T> {
    if (A.rows !== A.cols) throw new Error("solveJacobiMat: matrice non quadrata.");

    const { D, L, U } = decomposeDLU(A);
    const D_inv = inverseDiagonal(D);
    const LU    = L.add(U);

    // T = -D^{-1} * (L+U),  C = D^{-1} * b
    const T = D_inv.mul(LU).mul(-1);
    const C = D_inv.mul(b);

    let x     = A.like(A.rows, 1);
    let xNext = A.like(A.rows, 1);

    for (let k = 0; k < maxIter; k++) {
        xNext = T.mul(x).add(C);
        if (_hasConverged(x, xNext, tol)) return xNext;
        x = xNext;
    }
    return x;
}
