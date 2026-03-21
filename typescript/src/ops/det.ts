// ops/det.ts
//
// Ottimizzazioni:
//  1. classifyStructure: una sola O(n²) passata invece di tre.
//  2. Early-exit per zeri sulla diagonale.
//  3. det2x2 inline senza chiamate extra.
//
import { Matrix } from "..";
import { INumeric } from "../type";
import { lup } from "../decomposition";

/** Struttura matriciale rilevata in una singola passata O(n²). */
export type MatrixStructure = "zero" | "diagonal" | "lower" | "upper" | "dense";

/** Classifica la matrice in una passata sola. Esportata per riuso in hasProperty. */
export function classifyStructure<T extends INumeric<T>>(A: Matrix<T>, tol = 1e-10): MatrixStructure {
    const R = A.rows, C = A.cols;
    const d = A.data;
    let hasLower = false, hasUpper = false, hasDiag = false;

    for (let i = 0; i < R && !(hasLower && hasUpper); i++) {
        const off = i * C;
        for (let j = 0; j < C; j++) {
            if (!d[off + j].isNearZero(tol)) {
                if (i === j) hasDiag = true;
                else if (i > j) hasLower = true;
                else             hasUpper = true;
            }
        }
    }
    if (!hasLower && !hasUpper && !hasDiag) return "zero";
    if (!hasLower && !hasUpper)             return "diagonal";
    if (!hasUpper)                          return "lower";
    if (!hasLower)                          return "upper";
    return "dense";
}

export function det<T extends INumeric<T>>(A: Matrix<T>): T {
    if (!A.isSquare()) throw new Error("det: definito solo per matrici quadrate.");
    const n = A.rows;
    if (n === 1) return A.data[0];
    if (n === 2) return _det2x2(A);

    const struct = classifyStructure(A);
    if (struct === "zero")                           return A.zero;
    if (struct === "diagonal" || struct === "lower"
     || struct === "upper")                          return _detDiag(A);
    return _detLUP(A);
}

// ---- private ----

function _det2x2<T extends INumeric<T>>(A: Matrix<T>): T {
    const d = A.data;
    return d[0].multiply(d[3]).subtract(d[1].multiply(d[2]));
}

function _detDiag<T extends INumeric<T>>(A: Matrix<T>): T {
    const n = A.rows, C = A.cols;
    let result = A.one;
    for (let i = 0; i < n; i++) {
        const v = A.data[i * C + i];
        if (A.isZero(v)) return A.zero;
        result = result.multiply(v);
    }
    return result;
}

function _detLUP<T extends INumeric<T>>(A: Matrix<T>): T {
    const { U, swaps } = lup(A);
    let d = A.one;
    const n = U.rows;
    for (let i = 0; i < n; i++) {
        const v = U.data[i * n + i];
        if (A.isZero(v)) return A.zero;
        d = d.multiply(v);
    }
    return swaps & 1 ? d.negate() : d;
}
