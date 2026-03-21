// algoritm/inverse.ts
//
// smartInverse usa classifyStructure (una passata O(n²)) invece di cinque
// controlli strutturali separati. isOrthogonal (O(n³)) è rimosso dal
// percorso default — troppo costoso per una eurisica.
//
import { Matrix } from "..";
import { INumeric } from "../type";
import { solveLowerTriangular, solveUpperTriangular } from "../solver/triangular";
import { classifyStructure } from "../ops/det";

export function inverseDiagonal<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    const n = A.rows;
    const out = A.like(n, n);
    const ad = A.data, od = out.data;
    for (let i = 0; i < n; i++) {
        const v = ad[i * n + i];
        if (A.isZero(v)) throw new Error("inverseDiagonal: elemento diagonale nullo.");
        od[i * n + i] = A.one.divide(v);
    }
    return out;
}

export function inverse<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    return A.solve(A.likeIdentity(A.rows));
}

export function inverseOrthogonal<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    return A.t();
}

export function pseudoInverse<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    const At = A.t();
    return inverse(At.mul(A)).mul(At);
}

export function inverseTriangular<T extends INumeric<T>>(
    A: Matrix<T>, type: "upper" | "lower"
): Matrix<T> {
    const I = A.likeIdentity(A.rows);
    return type === "lower"
        ? solveLowerTriangular(A, I)
        : solveUpperTriangular(A, I);
}

/**
 * Sceglie l'algoritmo di inversione più efficiente in una sola O(n²) passata.
 * Non chiama isOrthogonal (O(n³)) per default: troppo costoso per un'euristica.
 * Usa inverseOrthogonal solo se il chiamante lo richiede esplicitamente
 * tramite A.t() dopo aver verificato la proprietà.
 */
export function smartInverse<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    if (!A.isSquare())  return pseudoInverse(A);

    const struct = classifyStructure(A);
    if (struct === "diagonal") return inverseDiagonal(A);
    if (struct === "upper")    return inverseTriangular(A, "upper");
    if (struct === "lower")    return inverseTriangular(A, "lower");

    // Caso generale: LUP
    return inverse(A);
}
