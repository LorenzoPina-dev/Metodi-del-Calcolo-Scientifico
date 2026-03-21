// ops/det.ts
import { Matrix } from "..";
import { INumeric } from "../type";
import { lup } from "../decomposition";

/**
 * Calcola il determinante della matrice. Restituisce T.
 * Usa LUP per il caso generale.
 */
export function det<T extends INumeric<T>>(A: Matrix<T>): T {
    if (!A.isSquare()) throw new Error("det: definito solo per matrici quadrate.");

    const n = A.rows;
    if (n === 1) return A.get(0, 0);
    if (n === 2) return det2x2(A);
    if (A.isDiagonal() || A.isUpperTriangular() || A.isLowerTriangular()) {
        return detTriangular(A);
    }
    return detLUP(A);
}

function det2x2<T extends INumeric<T>>(A: Matrix<T>): T {
    return A.get(0, 0).multiply(A.get(1, 1))
           .subtract(A.get(0, 1).multiply(A.get(1, 0)));
}

function detTriangular<T extends INumeric<T>>(A: Matrix<T>): T {
    let d = A.one;
    for (let i = 0; i < A.rows; i++) {
        const v = A.get(i, i);
        if (A.isZero(v)) return A.zero;
        d = d.multiply(v);
    }
    return d;
}

function detLUP<T extends INumeric<T>>(A: Matrix<T>): T {
    const { U, swaps } = lup(A);
    let d = A.one;
    for (let i = 0; i < U.rows; i++) {
        const v = U.get(i, i);
        if (A.isZero(v)) return A.zero;
        d = d.multiply(v);
    }
    // Segno dalla permutazione
    if (swaps % 2 !== 0) d = d.negate();
    return d;
}
