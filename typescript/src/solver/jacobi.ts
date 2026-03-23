// solver/jacobi.ts — soglia WASM calibrata (bassa: loop intero in WASM)
import { Matrix }        from "..";
import { Float64M, INumeric } from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";
import { _hasConverged } from "./_hasConverged";

export function solveJacobiMat<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>,
    tol = 1e-10, maxIter = 5000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveJacobiMat: matrice non quadrata.");

    if (A.isFloat64 && n * n >= WASM_THRESHOLD.ITERATIVE) {
        const w = getBridgeSync();
        if (w) {
            const aPtr       = w.writeFloat64M(A.data as any);
            const bPtr       = w.writeFloat64M(b.data as any);
            const xPtr       = w.allocOutput(n);
            const xNewPtr    = w.allocOutput(n);
            const diagInvPtr = w.allocOutput(n);
            w.exports.jacobiSolve(aPtr, bPtr, xPtr, xNewPtr, diagInvPtr, n, tol, maxIter);
            const flat = w.readF64(xPtr, n);
            w.reset();
            const out = A.like(n, 1);
            for (let i=0;i<n;i++) out.data[i] = A.zero.fromNumber(flat[i]);
            return out;
        }
        return _jacobiF64(A as any, b as any, tol, maxIter);
    }
    return _jacobiGeneric(A, b, tol, maxIter);
}

function _jacobiF64(A: Matrix<Float64M>, b: Matrix<Float64M>, tol: number, maxIter: number): Matrix<any> {
    const n = A.rows, ad = A.data, bd = b.data;
    const a_raw = new Float64Array(n*n), b_raw = new Float64Array(n), diagInv = new Float64Array(n);
    for (let i=0;i<n*n;i++) a_raw[i] = ad[i].value;
    for (let i=0;i<n;i++)   b_raw[i] = bd[i].value;
    for (let i=0;i<n;i++) {
        const d = a_raw[i*n+i];
        if (Math.abs(d) < 1e-300) throw new Error(`Jacobi: pivot nullo alla riga ${i+1}.`);
        diagInv[i] = 1.0 / d;
    }
    let x = new Float64Array(n), xNew = new Float64Array(n);
    for (let iter=0;iter<maxIter;iter++) {
        let maxDiff=0.0, maxAbsX=0.0;
        for (let i=0;i<n;i++) {
            const off=i*n; let s=0.0;
            for (let j=0;j<i;j++) s += a_raw[off+j] * x[j];
            for (let j=i+1;j<n;j++) s += a_raw[off+j] * x[j];
            const xi = (b_raw[i] - s) * diagInv[i];
            xNew[i] = xi;
            const diff = xi - x[i]; const absDiff = diff<0?-diff:diff;
            if (absDiff > maxDiff) maxDiff = absDiff;
            const ax = xi<0?-xi:xi; if (ax > maxAbsX) maxAbsX = ax;
        }
        const tmp = x; x = xNew; xNew = tmp;
        const denom = maxAbsX > 1.0 ? maxAbsX : 1.0;
        if (maxDiff / denom < tol) break;
    }
    const out = A.like(n, 1);
    for (let i=0;i<n;i++) out.data[i] = A.zero.fromNumber(x[i]);
    return out;
}

function _jacobiGeneric<T extends INumeric<T>>(A: Matrix<T>, b: Matrix<T>, tol: number, maxIter: number): Matrix<T> {
    const n = A.rows, ad = A.data, bd = b.data;
    let x = A.like(n, 1);
    for (let iter=0;iter<maxIter;iter++) {
        const xNext = A.like(n, 1); const nd=xNext.data, xd=x.data;
        for (let i=0;i<n;i++) {
            const off=i*n; let s=bd[i];
            for (let j=0;j<i;j++) s=s.subtract(ad[off+j].multiply(xd[j]));
            for (let j=i+1;j<n;j++) s=s.subtract(ad[off+j].multiply(xd[j]));
            nd[i] = s.divide(ad[off+i]);
        }
        if (_hasConverged(x, xNext, tol)) return xNext;
        x = xNext;
    }
    return x;
}
