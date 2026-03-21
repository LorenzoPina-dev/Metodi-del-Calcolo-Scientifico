// solver/solve.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function solve<T extends INumeric<T>>(
    this: Matrix<T>,
    b: Matrix<T>,
    method: string = "LUP"
): Matrix<T> {
    switch (method.toUpperCase()) {

        case "LU": {
            const { L, U } = Matrix.decomp.lu(this);
            const y = Matrix.solver.solveLowerTriangular(L, b);
            return Matrix.solver.solveUpperTriangular(U, y);
        }

        case "LUP": {
            const { L, U, P } = Matrix.decomp.lup(this);
            const bp = applyPermutation(b, P);
            const y  = Matrix.solver.solveLowerTriangular(L, bp);
            return Matrix.solver.solveUpperTriangular(U, y);
        }

        case "CHOLESKY": {
            const { L } = Matrix.decomp.cholesky(this);
            const y  = Matrix.solver.solveLowerTriangular(L, b);
            const Lt = L.t();
            return Matrix.solver.solveUpperTriangular(Lt, y);
        }

        case "QR": {
            const { Q, R } = Matrix.decomp.qr(this);
            const Qtb = Q.t().mul(b);
            return Matrix.solver.solveUpperTriangular(R, Qtb);
        }

        case "JACOBI":
            return Matrix.solver.solveJacobiMat(this, b);

        case "GAUSS-SEIDEL":
            return Matrix.solver.solveGaussSeidelMat(this, b);

        default:
            throw new Error(`solve: metodo '${method}' non riconosciuto.`);
    }
}

/** Riordina le righe di b secondo il vettore di permutazione P. */
function applyPermutation<T extends INumeric<T>>(b: Matrix<T>, P: number[]): Matrix<T> {
    if (P.length !== b.rows) throw new Error("applyPermutation: dimensione di P non coincide con b.");
    const result = b.like(b.rows, b.cols);
    for (let i = 0; i < b.rows; i++) {
        for (let j = 0; j < b.cols; j++) {
            result.set(i, j, b.get(P[i], j));
        }
    }
    return result;
}
