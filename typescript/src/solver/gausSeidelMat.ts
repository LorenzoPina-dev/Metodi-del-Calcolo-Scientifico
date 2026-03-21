// solver/gausSeidelMat.ts
import { Matrix } from "..";
import { INumeric } from "../type";
import { decomposeDLU } from "../decomposition/dlu";
import { _hasConverged } from "./_hasConverged";

/**
 * Risolve A*x = b con il metodo di Gauss-Seidel (forma matriciale).
 * Formula: x_{k+1} = (D+L)^{-1} * (b - U * x_k)
 */
export function solveGaussSeidelMat<T extends INumeric<T>>(
    A: Matrix<T>,
    b: Matrix<T>,
    tol: number = 1e-10,
    maxIter: number = 1000
): Matrix<T> {
    if (A.rows !== A.cols) throw new Error("solveGaussSeidelMat: matrice non quadrata.");

    const { D, L, U } = decomposeDLU(A);
    const DL     = D.add(L);
    const DL_inv = DL.inverse();       // usa smartInverse → solveLowerTriangular

    // T = -(D+L)^{-1} * U,  C = (D+L)^{-1} * b
    const T = DL_inv.mul(U).mul(-1);
    const C = DL_inv.mul(b);

    let x     = A.like(A.rows, 1);
    let xNext = A.like(A.rows, 1);

    for (let k = 0; k < maxIter; k++) {
        xNext = T.mul(x).add(C);
        if (_hasConverged(x, xNext, tol)) return xNext;
        x = xNext;
    }
    return x;
}
