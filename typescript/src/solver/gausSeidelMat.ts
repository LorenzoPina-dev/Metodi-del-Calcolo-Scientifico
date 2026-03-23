// solver/gausSeidelMat.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";
import { _hasConverged } from "./_hasConverged";

export function solveGaussSeidelMat<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>,
    tol = 1e-10, maxIter = 1000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveGaussSeidelMat: matrice non quadrata.");

    if (A.isFloat64 && n * n >= WASM_THRESHOLD.ITERATIVE) {
        const w = getBridgeSync();
        if (w) {
            const aPtr = w.writeFloat64M(A.data as any);
            const bPtr = w.writeFloat64M(b.data as any);
            const xPtr = w.allocOutput(n);
            w.exports.gaussSeidelSolve(aPtr, bPtr, xPtr, n, tol, maxIter);
            const flat = w.readF64(xPtr, n);
            w.reset();
            const out = A.like(n, 1);
            for (let i=0;i<n;i++) out.data[i] = A.zero.fromNumber(flat[i]);
            return out;
        }
    }

    const ad = A.data, bd = b.data;
    let x = A.like(n, 1);
    for (let iter=0;iter<maxIter;iter++) {
        const xPrev = x.clone(); const xd = x.data;
        for (let i=0;i<n;i++) {
            const off=i*n; let s=bd[i];
            for (let j=0;j<i;j++) s=s.subtract(ad[off+j].multiply(xd[j]));
            for (let j=i+1;j<n;j++) s=s.subtract(ad[off+j].multiply(xd[j]));
            xd[i] = s.divide(ad[off+i]);
        }
        if (_hasConverged(xPrev, x, tol)) return x;
    }
    return x;
}
