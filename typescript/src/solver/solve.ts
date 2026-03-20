import { Matrix } from "..";

export function solve(this: Matrix, b: Matrix, method: string = 'LUP'): Matrix {
    switch (method.toUpperCase()) {
        case 'LU':{
            // 1. Decomposizione: A = L * U
            const { L, U } = Matrix.decomp.lu(this);
            // 2. Risoluzione L * y = b
            const y = Matrix.solver.solveLowerTriangular(L, b);
            // 3. Risoluzione U * x = y
            return Matrix.solver.solveUpperTriangular(U, y);
        }
        case 'LUP': {
            // 1. Decomposizione: A = P^T * L * U
            const { L, U, P } = Matrix.decomp.lup(this);
            // 2. Permutazione: b_perm = P * b
            const b_perm = applyPermutation(b, P);
            // 3. Risoluzione L * y = b_perm
            const y = Matrix.solver.solveLowerTriangular(L, b_perm);
            // 4. Risoluzione U * x = y
            return Matrix.solver.solveUpperTriangular(U, y);
        }
        case 'CHOLESKY': {
            // 1. Decomposizione: A = L * L^T
            const {L} = Matrix.decomp.cholesky(this);
            // 2. Risoluzione L * y = b
            const y = Matrix.solver.solveLowerTriangular(L, b);
            // 3. Risoluzione L^T * x = y
            const LT = L.t();
            return Matrix.solver.solveUpperTriangular(LT, y);
        }

        case 'QR': {
            // 1. Decomposizione: A = Q * R
            const { Q, R } = Matrix.decomp.qr(this);
            // 2. Calcolo Q^T * b (Q è ortogonale, quindi Q^-1 = Q^T)
            const Qtb = Q.t().mul(b);
            // 3. Risoluzione R * x = Q^T * b (R è sempre triangolare superiore)
            return Matrix.solver.solveUpperTriangular(R, Qtb);
        }

        case 'JACOBI':
            // Jacobi è iterativo, non usa solveTriangular ma un ciclo
            return Matrix.solver.solveJacobiMat(this, b);
        case 'GAUSS-SEIDEL':
            // Gauss-Seidel è iterativo, non usa solveTriangular ma un ciclo
            return Matrix.solver.solveGaussSeidelMat(this, b);
        default:
            throw new Error(`Metodo ${method} non riconosciuto.`);
    }
}
/**
 * Applica la permutazione P al vettore (o matrice) b
 * @param P Vettore degli indici permutati (es. [2, 0, 1])
 * @returns Una nuova Matrix con le righe riordinate
 */
function applyPermutation(b:Matrix,P: number[]): Matrix {
    if (P.length !== b.rows) {
        throw new Error("applyPermutation: dimension mismatch");
    }

    // Creiamo una nuova matrice delle stesse dimensioni di b
    const result = Matrix.zeros(b.rows, b.cols);

    for (let i = 0; i < b.rows; i++) {
        const sourceRow = P[i];
        for (let j = 0; j < b.cols; j++) {
            // Copiamo il valore dalla riga sorgente indicata da P
            result.set(i, j, b.get(sourceRow, j));
        }
    }

    return result;
}