// decomposition/ldlt.ts — con fast-path WASM
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync } from "../wasm/wasm_bridge";

export function ldlt<T extends INumeric<T>>(A: Matrix<T>): { L: Matrix<T>; D: Matrix<T> } {
    if (A.rows !== A.cols) throw new Error("ldlt: matrice non quadrata.");
    if (!A.isSymmetric())  throw new Error("ldlt: matrice non simmetrica.");
    const n = A.rows;

    if (A.isFloat64) {
        const w = getBridgeSync();
        if (w) {
            const aPtr = w.writeFloat64M(A.data as any);
            const lPtr = w.allocOutput(n * n);
            const dPtr = w.allocOutput(n);          // solo la diagonale
            const ret  = w.exports.ldltDecomp(aPtr, lPtr, dPtr, n);
            if (ret < 0) { w.reset(); throw new Error("ldlt: pivot nullo."); }
            const lFlat = w.readF64(lPtr, n * n);
            const dFlat = w.readF64(dPtr, n);
            w.reset();
            const L = A.like(n, n), D = A.like(n, n);
            for (let i = 0; i < n*n; i++) L.data[i] = A.zero.fromNumber(lFlat[i]);
            for (let i = 0; i < n; i++) D.data[i*n+i] = A.zero.fromNumber(dFlat[i]);
            return { L, D };
        }
    }

    // ── TS fallback ────────────────────────────────────────────
    const L = A.likeIdentity(n), D = A.like(n, n);
    const ad = A.data, ld = L.data, dd = D.data;
    for (let j = 0; j < n; j++) {
        let djj = ad[j*n+j];
        for (let k = 0; k < j; k++) { const ljk = ld[j*n+k]; djj = djj.subtract(ljk.multiply(ljk).multiply(dd[k*n+k])); }
        if (A.isZero(djj)) throw new Error(`ldlt: pivot nullo alla colonna ${j}.`);
        dd[j*n+j] = djj;
        for (let i = j+1; i < n; i++) {
            let lij = ad[i*n+j];
            for (let k = 0; k < j; k++) lij = lij.subtract(ld[i*n+k].multiply(ld[j*n+k]).multiply(dd[k*n+k]));
            ld[i*n+j] = lij.divide(djj);
        }
    }
    return { L, D };
}

export function solveLDLT<T extends INumeric<T>>(A: Matrix<T>, b: Matrix<T>): Matrix<T> {
    const { L, D } = ldlt(A);
    const n = A.rows;
    const ld = L.data, dd = D.data, bd = b.data;

    // L*y = b (sostituzione avanti, L unit.)
    const y = b.like(n, 1); const yd = y.data;
    yd[0] = bd[0];
    for (let i = 1; i < n; i++) {
        let s = bd[i];
        for (let k = 0; k < i; k++) s = s.subtract(ld[i*n+k].multiply(yd[k]));
        yd[i] = s;
    }
    // D*z = y
    const z = b.like(n, 1); const zd = z.data;
    for (let i = 0; i < n; i++) zd[i] = yd[i].divide(dd[i*n+i]);
    // L^T*x = z
    const x = b.like(n, 1); const xd = x.data;
    xd[n-1] = zd[n-1];
    for (let i = n-2; i >= 0; i--) {
        let s = zd[i];
        for (let k = i+1; k < n; k++) s = s.subtract(ld[k*n+i].multiply(xd[k]));
        xd[i] = s;
    }
    return x;
}
