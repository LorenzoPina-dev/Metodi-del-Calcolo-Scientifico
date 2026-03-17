import { Matrix } from "../core";
import { identity } from "../init";

export function luPivotingTotal(A: Matrix): { P: Matrix; L: Matrix; U: Matrix } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("Matrix must be square");
    let A_old = A.clone(), M = identity(n), P = identity(n);
    for (let col = 0; col < n - 1; col++) {
        let maxVal = 0, pivotRow = col;
        for (let i = col; i < n; i++) {
            const val = Math.abs(A_old.get(i, col));
            if (val > maxVal) { maxVal = val; pivotRow = i; }
        }
        if (Matrix.isZero(maxVal)) continue;
        if (pivotRow !== col) {
            const Pn = identity(n);
            Pn.set(col, col, 0); Pn.set(pivotRow, pivotRow, 0);
            Pn.set(col, pivotRow, 1); Pn.set(pivotRow, col, 1);
            P = Pn.multiply(P); A_old = Pn.multiply(A_old);
        }
        const Ltilden = identity(n);
        for (let i = col + 1; i < n; i++) Ltilden.set(i, col, -A_old.get(i, col) / A_old.get(col, col));
        M = Ltilden.multiply(M); A_old = Ltilden.multiply(A_old);
    }
    const U = A_old;
    const L = P.multiply(M.inverse());
    return { P, L, U };
}