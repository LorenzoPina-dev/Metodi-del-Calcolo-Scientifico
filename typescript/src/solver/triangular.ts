// solver/triangular.ts
//
// Sostituzione in avanti/indietro con accesso diretto a data[].
// Nessuna chiamata a get/set nel loop interno.
//
import { Matrix } from "..";
import { INumeric } from "../type";

export function solveLowerTriangular<T extends INumeric<T>>(
    L: Matrix<T>, b: Matrix<T>
): Matrix<T> {
    const N = L.rows;
    if (N !== L.cols) throw new Error("solveLowerTriangular: L deve essere quadrata.");

    const x = b.like(N, b.cols);
    const ld = L.data, bd = b.data, xd = x.data;
    const BC = b.cols;

    for (let col = 0; col < BC; col++) {
        // x[0] = b[0] / L[0,0]
        xd[col] = bd[col].divide(ld[0]);

        for (let i = 1; i < N; i++) {
            let s = bd[i * BC + col];
            for (let k = 0; k < i; k++) {
                s = s.subtract(ld[i * N + k].multiply(xd[k * BC + col]));
            }
            xd[i * BC + col] = s.divide(ld[i * N + i]);
        }
    }
    return x;
}

export function solveUpperTriangular<T extends INumeric<T>>(
    U: Matrix<T>, b: Matrix<T>
): Matrix<T> {
    const N = U.rows;
    if (N !== U.cols) throw new Error("solveUpperTriangular: U deve essere quadrata.");

    const x = b.like(N, b.cols);
    const ud = U.data, bd = b.data, xd = x.data;
    const BC = b.cols;

    for (let col = 0; col < BC; col++) {
        // x[N-1] = b[N-1] / U[N-1, N-1]
        xd[(N - 1) * BC + col] = bd[(N - 1) * BC + col].divide(ud[(N - 1) * N + (N - 1)]);

        for (let i = N - 2; i >= 0; i--) {
            let s = bd[i * BC + col];
            for (let k = i + 1; k < N; k++) {
                s = s.subtract(ud[i * N + k].multiply(xd[k * BC + col]));
            }
            xd[i * BC + col] = s.divide(ud[i * N + i]);
        }
    }
    return x;
}
