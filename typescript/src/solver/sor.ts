// solver/sor.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";
import { _hasConverged } from "./_hasConverged";

export function solveSOR<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>,
    omega = 1.5, tol = 1e-10, maxIter = 2000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveSOR: matrice non quadrata.");
    if (omega <= 0 || omega >= 2) throw new Error(`solveSOR: ω deve essere in (0, 2).`);

    if (A.isFloat64 && n * n >= WASM_THRESHOLD.ITERATIVE) {
        const w = getBridgeSync();
        if (w) {
            const aPtr = w.writeFloat64M(A.data as any);
            const bPtr = w.writeFloat64M(b.data as any);
            const xPtr = w.allocOutput(n);
            w.exports.sorSolve(aPtr, bPtr, xPtr, n, omega, tol, maxIter);
            const flat = w.readF64(xPtr, n);
            w.reset();
            const out = A.like(n, 1);
            for (let i=0;i<n;i++) out.data[i] = A.zero.fromNumber(flat[i]);
            return out;
        }
    }

    const ad = A.data, bd = b.data;
    const omegaT = A.zero.fromNumber(omega), oneMinusOmegaT = A.zero.fromNumber(1-omega);
    let x = A.like(n, 1);
    for (let iter=0;iter<maxIter;iter++) {
        const xPrev = x.clone(); const xd = x.data;
        for (let i=0;i<n;i++) {
            const off=i*n; let s=bd[i];
            for (let j=0;j<i;j++) s=s.subtract(ad[off+j].multiply(xd[j]));
            for (let j=i+1;j<n;j++) s=s.subtract(ad[off+j].multiply(xd[j]));
            xd[i] = oneMinusOmegaT.multiply(xd[i]).add(omegaT.multiply(s.divide(ad[off+i])));
        }
        if (_hasConverged(xPrev, x, tol)) return x;
    }
    return x;
}
