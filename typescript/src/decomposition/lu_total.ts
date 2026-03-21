import { Matrix } from "..";
import { Float64M } from "../type";

export function luPivoting(A: Matrix): { P: Matrix; L: Matrix; U: Matrix } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("Matrix must be square");

    const U = A.clone();
    const L = Matrix.identity(n);
    const P = Matrix.identity(n);

    const EPS = new Float64M(1e-12);

    for (let k = 0; k < n; k++) {
        // Pivoting (parziale)
        let pivotRow = k;
        let max = U.get(k, k).abs();

        for (let i = k + 1; i < n; i++) {
            const val = U.get(i, k).abs();
            if (val.greaterThan(max)) {
                max = val;
                pivotRow = i;
            }
        }

        if (max.lessThan(EPS)) throw new Error("Matrix is singular or nearly singular");

        // Swap righe
        if (pivotRow !== k) {
            swapRows.call(U, k, pivotRow);
            swapRows.call(P, k, pivotRow);

            // swap solo la parte già costruita di L
            for (let j = 0; j < k; j++) {
                const temp = L.get(k, j);
                L.set(k, j, L.get(pivotRow, j));
                L.set(pivotRow, j, temp);
            }
        }

        // Eliminazione
        for (let i = k + 1; i < n; i++) {
            const factor = U.get(i, k).divide(U.get(k, k));
            L.set(i, k, factor);

            for (let j = k; j < n; j++) {
                U.set(i, j, U.get(i, j).subtract(factor.multiply(U.get(k, j))));
            }
        }
    }

    return { P, L, U };
}
function swapRows(this: Matrix, i: number, j: number) {
    for (let col = 0; col < this.cols; col++) {
        const temp = this.get(i, col);
        this.set(i, col, this.get(j, col));
        this.set(j, col, temp);
    }
}