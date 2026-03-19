import { Matrix } from "..";
import { decomposeDLU } from "../decomposition/dlu";
import { _hasConverged } from "./_hasConverged";

/**
 * Risolve Ax = b usando il metodo di Gauss-Seidel in forma puramente matriciale.
 * Formula: x_next = (D + L)^-1 * (b - U * x)
 */
export function solveGaussSeidelMat(A:Matrix, b: Matrix, tol: number = Matrix.EPS, maxIter: number = 1000): Matrix {
    if (A.rows !== A.cols) throw new Error("Matrice non quadrata");

    const { D, L, U } = decomposeDLU(A);

    // D + L
    const DL = D.add(L);
    
    // Calcolo di (D + L)^-1 usando il tuo metodo inverse() esistente
    const DL_inv = DL.inverse();

    // Costruzione della matrice di iterazione T e del vettore C
    // T = - (D + L)^-1 * U
    const T = DL_inv.mul(U).mul(-1);
    // C = (D + L)^-1 * b
    const C = DL_inv.mul(b);

    let x = new Matrix(A.rows, 1);
    let xNext = new Matrix(A.rows, 1);

    for (let k = 1; k <= maxIter; k++) {
        // x_next = T * x + C
        xNext = T.mul(x).add(C);

        if (_hasConverged(x, xNext, tol)) return xNext;
        x = xNext;
    }

    console.warn("Gauss-Seidel (Matriciale): Raggiunto il numero massimo di iterazioni.");
    return x;
}