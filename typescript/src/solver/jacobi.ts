import { inverseDiagonal } from "../algoritm/inverse";
import { Matrix } from "..";
import { decomposeDLU } from "../decomposition/dlu";
import { _hasConverged } from "./_hasConverged";

/**
 * Risolve Ax = b usando il metodo di Jacobi in forma puramente matriciale.
 * Formula: x_next = D^-1 * (b - (L + U) * x)
 */
export function solveJacobiMat(A:Matrix, b: Matrix, tol: number = Matrix.EPS, maxIter: number = 1000): Matrix {
    if (A.rows !== A.cols) throw new Error("Matrice non quadrata");
    
    const { D, L, U } = decomposeDLU(A);
    
    // Calcolo di D^-1 (Inversa di una matrice diagonale: reciproco degli elementi)
    const D_inv = inverseDiagonal(D);

    // L + U
    const LU = L.add(U);

    // Costruzione della matrice di iterazione T e del vettore C
    // T = - D^-1 * (L + U)
    const T = D_inv.mul(LU).mul(-1);
    // C = D^-1 * b
    const C = D_inv.mul(b);

    let x = new Matrix(A.rows, 1);
    let xNext = new Matrix(A.rows, 1);

    for (let k = 1; k <= maxIter; k++) {
        // x_next = T * x + C
        xNext = T.mul(x).add(C);

        if (_hasConverged(x, xNext, tol)) return xNext;
        x = xNext; // Nessun bisogno di clone(), l'assegnazione va bene perché T.multiply crea una nuova matrice
    }
    
    return x;
}