// ops/norm.ts
import { Matrix } from "..";
import { INumeric } from "../type";

type NormType = "1" | "2" | "inf" | "fro";

/**
 * Norma di una matrice o vettore.
 * Restituisce sempre number (la norma è sempre reale e positiva).
 * Per tipi complessi usa il modulo di ogni elemento (toNumber()).
 */
export function norm<T extends INumeric<T>>(this: Matrix<T>, type: NormType = "2"): number {
    const isVec = this.rows === 1 || this.cols === 1;
    return isVec ? normVector(this, type) : normMatrix(this, type);
}

function normVector<T extends INumeric<T>>(A: Matrix<T>, type: NormType): number {
    const data = A.data;
    const len = data.length;

    switch (type.toUpperCase()) {
        case "1": {
            let s = 0;
            for (let i = 0; i < len; i++) s += data[i].abs().toNumber();
            return s;
        }
        case "INF": {
            let max = 0;
            for (let i = 0; i < len; i++) {
                const v = data[i].abs().toNumber();
                if (v > max) max = v;
            }
            return max;
        }
        case "2":
        case "FRO": {
            let sumSq = 0;
            for (let i = 0; i < len; i++) {
                const v = data[i].abs().toNumber();
                sumSq += v * v;
            }
            return Math.sqrt(sumSq);
        }
        default:
            throw new Error(`norm: tipo '${type}' non supportato per vettori.`);
    }
}

function normMatrix<T extends INumeric<T>>(A: Matrix<T>, type: NormType): number {
    switch (type.toUpperCase()) {
        case "1":   return norm1Matrix(A);
        case "INF": return normInfMatrix(A);
        case "FRO": return normFrobenius(A);
        case "2":
            throw new Error("norm 2 matriciale richiede SVD (non ancora implementato).");
        default:
            throw new Error(`norm: tipo '${type}' non supportato per matrici.`);
    }
}

function norm1Matrix<T extends INumeric<T>>(A: Matrix<T>): number {
    let max = 0;
    for (let j = 0; j < A.cols; j++) {
        let s = 0;
        for (let i = 0; i < A.rows; i++) s += A.get(i, j).abs().toNumber();
        if (s > max) max = s;
    }
    return max;
}

function normInfMatrix<T extends INumeric<T>>(A: Matrix<T>): number {
    let max = 0;
    for (let i = 0; i < A.rows; i++) {
        let s = 0;
        for (let j = 0; j < A.cols; j++) s += A.get(i, j).abs().toNumber();
        if (s > max) max = s;
    }
    return max;
}

function normFrobenius<T extends INumeric<T>>(A: Matrix<T>): number {
    let sumSq = 0;
    for (let i = 0; i < A.data.length; i++) {
        const v = A.data[i].abs().toNumber();
        sumSq += v * v;
    }
    return Math.sqrt(sumSq);
}
