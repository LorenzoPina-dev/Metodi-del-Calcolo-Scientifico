import { Matrix } from "../core";
import { zeros } from "../init";

export function solveLowerTriangular(L: Matrix, b: Matrix): Matrix {
    const N = L.rows;
    if (N !== L.cols) throw new Error("Matrix L must be square");
    if (L.subtract(L.tril()).totalSum() > 1e-15) throw new Error("Matrix L is not lower triangular");
    const x = zeros(N, b.cols);
    for (let j = 0; j < b.cols; j++) {
        x.set(0, j, b.get(0, j) / L.get(0, 0));
        for (let i = 1; i < N; i++) {
            let sum = 0;
            for (let k = 0; k < i; k++) sum += L.get(i, k) * x.get(k, j);
            x.set(i, j, (b.get(i, j) - sum) / L.get(i, i));
        }
    }
    return x;
}
export function solveUpperTriangular(U: Matrix, b: Matrix): Matrix { 
    const N = U.rows;
        if (N !== U.cols) throw new Error("Matrix U must be square");
        if (U.subtract(U.triu()).totalSum() > 1e-15) throw new Error("Matrix U is not upper triangular");
        const x = zeros(N, b.cols);
        for (let j = 0; j < b.cols; j++) {
            x.set(N - 1, j, b.get(N - 1, j) / U.get(N - 1, N - 1));
            for (let i = N - 2; i >= 0; i--) {
                let sum = 0;
                for (let k = i + 1; k < N; k++) sum += U.get(i, k) * x.get(k, j);
                x.set(i, j, (b.get(i, j) - sum) / U.get(i, i));
            }
        }
        return x;
}