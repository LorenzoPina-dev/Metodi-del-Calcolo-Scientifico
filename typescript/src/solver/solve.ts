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
            return Matrix.solver.solveUpperTriangular(L.ct(), y);
        }

        case "LDLT":
            return Matrix.decomp.solveLDLT(this, b);

        case "QR": {
            const { Q, R } = Matrix.decomp.qr(this);
            return Matrix.solver.solveUpperTriangular(R, Q.ct().mul(b));
        }

        case "JACOBI":
            return Matrix.solver.solveJacobiMat(this, b);

        case "GAUSS-SEIDEL":
            return Matrix.solver.solveGaussSeidelMat(this, b);

        case "SOR":
            // Default ω=1.5 — usa solveSOR() direttamente per un valore custom
            return Matrix.solver.solveSOR(this, b, 1.5);

        case "JOR":
            // Default ω=1.0 (= Jacobi classico) — usa solveJorMat() direttamente per omega custom
            return Matrix.solver.solveJorMat(this, b, 1.0);

        case "CG":
        case "CONJUGATE-GRADIENT":
            return Matrix.solver.solveCG(this, b);

        default:
            throw new Error(
                `solve: metodo '${method}' non riconosciuto. ` +
                `Disponibili: LU, LUP, QR, CHOLESKY, LDLT, JACOBI, GAUSS-SEIDEL, SOR, JOR, CG`
            );
    }
}

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
