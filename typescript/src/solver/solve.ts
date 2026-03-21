// solver/solve.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function solve<T extends INumeric<T>>(
    this: Matrix<T>, b: Matrix<T>, method = "LUP"
): Matrix<T> {
    switch (method.toUpperCase()) {

        case "LU": {
            const { L, U } = Matrix.decomp.lu(this);
            return Matrix.solver.solveUpperTriangular(U,
                   Matrix.solver.solveLowerTriangular(L, b));
        }

        case "LUP": {
            const { L, U, P } = Matrix.decomp.lup(this);
            const bp = _permute(b, P);
            return Matrix.solver.solveUpperTriangular(U,
                   Matrix.solver.solveLowerTriangular(L, bp));
        }

        case "CHOLESKY": {
            const { L } = Matrix.decomp.cholesky(this);
            const y = Matrix.solver.solveLowerTriangular(L, b);
            // L^H (trasposta coniugata di L)
            return Matrix.solver.solveUpperTriangular(L.ct(), y);
        }

        case "QR": {
            const { Q, R } = Matrix.decomp.qr(this);
            //
            // A = Q*R  →  R*x = Q^H * b
            //
            // Q^H è la trasposta coniugata (adjoint):
            //   - Per matrici reali  (Float64M, Rational): Q^H = Q^T = Q^{-1}
            //   - Per matrici complesse: Q^H = conj(Q)^T = Q^{-1}
            //
            // SBAGLIATO per complessi:  Q.t()   (trasposta semplice)
            // CORRETTO per tutti i tipi: Q.ct()  (trasposta coniugata)
            //
            return Matrix.solver.solveUpperTriangular(R, Q.ct().mul(b));
        }

        case "JACOBI":
            return Matrix.solver.solveJacobiMat(this, b);

        case "GAUSS-SEIDEL":
            return Matrix.solver.solveGaussSeidelMat(this, b);

        default:
            throw new Error(`solve: metodo '${method}' non riconosciuto.`);
    }
}

/** Riordina le righe di b secondo P (P[i] = riga sorgente per la riga i). */
function _permute<T extends INumeric<T>>(b: Matrix<T>, P: number[]): Matrix<T> {
    const n = b.rows, BC = b.cols;
    const out = b.like(n, BC);
    const bd = b.data, od = out.data;
    for (let i = 0; i < n; i++) {
        const src = P[i] * BC, dst = i * BC;
        for (let j = 0; j < BC; j++) od[dst + j] = bd[src + j];
    }
    return out;
}
