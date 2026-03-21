// algoritm/inverse.ts
import { Matrix } from "..";
import { INumeric } from "../type";
import { solveLowerTriangular, solveUpperTriangular } from "../solver/triangular";

/**
 * Inversa di una matrice diagonale: reciproco di ogni elemento diagonale.
 */
export function inverseDiagonal<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    const out = A.like(A.rows, A.cols);
    for (let i = 0; i < A.rows; i++) {
        const v = A.get(i, i);
        if (A.isZero(v)) throw new Error("inverseDiagonal: elemento diagonale nullo.");
        out.set(i, i, A.one.divide(v));
    }
    return out;
}

/**
 * Inversa generale tramite LUP (risolve A*X = I colonna per colonna).
 */
export function inverse<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    return A.solve(A.likeIdentity(A.rows));
}

/** Inversa di una matrice ortogonale: A^{-1} = A^T. */
export function inverseOrthogonal<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    return A.t();
}

/** Pseudo-inversa di Moore-Penrose: (A^T A)^{-1} A^T. */
export function pseudoInverse<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    const At = A.t();
    return inverse(At.mul(A)).mul(At);
}

/**
 * Inversa di una matrice triangolare (superiore o inferiore).
 * Risolve A * X = I sfruttando la struttura triangolare.
 */
export function inverseTriangular<T extends INumeric<T>>(
    A: Matrix<T>,
    type: "upper" | "lower"
): Matrix<T> {
    const I  = A.likeIdentity(A.rows);
    if (type === "lower") return solveLowerTriangular(A, I);
    return solveUpperTriangular(A, I);
}

/**
 * Sceglie automaticamente l'algoritmo di inversione più efficiente.
 */
export function smartInverse<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    if (!A.isSquare())          return pseudoInverse(A);
    if (A.isDiagonal())         return inverseDiagonal(A);
    if (A.isUpperTriangular())  return inverseTriangular(A, "upper");
    if (A.isLowerTriangular())  return inverseTriangular(A, "lower");
    if (A.isOrthogonal())       return inverseOrthogonal(A);
    return inverse(A);
}
