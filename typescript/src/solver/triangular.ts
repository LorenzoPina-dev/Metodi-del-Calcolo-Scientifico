// solver/triangular.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Risolve L * x = b (sostituzione in avanti).
 * L deve essere triangolare inferiore quadrata.
 */
export function solveLowerTriangular<T extends INumeric<T>>(
    L: Matrix<T>,
    b: Matrix<T>
): Matrix<T> {
    const N = L.rows;
    if (N !== L.cols) throw new Error("solveLowerTriangular: L deve essere quadrata.");

    const x = b.like(N, b.cols);

    for (let col = 0; col < b.cols; col++) {
        // x[0] = b[0] / L[0,0]
        x.set(0, col, b.get(0, col).divide(L.get(0, 0)));

        for (let i = 1; i < N; i++) {
            let s = b.get(i, col);
            for (let k = 0; k < i; k++) {
                s = s.subtract(L.get(i, k).multiply(x.get(k, col)));
            }
            x.set(i, col, s.divide(L.get(i, i)));
        }
    }
    return x;
}

/**
 * Risolve U * x = b (sostituzione all'indietro).
 * U deve essere triangolare superiore quadrata.
 */
export function solveUpperTriangular<T extends INumeric<T>>(
    U: Matrix<T>,
    b: Matrix<T>
): Matrix<T> {
    const N = U.rows;
    if (N !== U.cols) throw new Error("solveUpperTriangular: U deve essere quadrata.");

    const x = b.like(N, b.cols);

    for (let col = 0; col < b.cols; col++) {
        // x[N-1] = b[N-1] / U[N-1, N-1]
        x.set(N - 1, col, b.get(N - 1, col).divide(U.get(N - 1, N - 1)));

        for (let i = N - 2; i >= 0; i--) {
            let s = b.get(i, col);
            for (let k = i + 1; k < N; k++) {
                s = s.subtract(U.get(i, k).multiply(x.get(k, col)));
            }
            x.set(i, col, s.divide(U.get(i, i)));
        }
    }
    return x;
}
