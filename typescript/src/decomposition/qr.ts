// decomposition/qr.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

export function qr<T extends INumeric<T>>(A: Matrix<T>): { Q: Matrix<T>; R: Matrix<T> } {
    const m = A.rows, n = A.cols;

    if (A.isFloat64 && m * n >= WASM_THRESHOLD.DECOMP) {
        const w = getBridgeSync();
        if (w) {
            const wPtr = w.writeFloat64M(A.data as any);
            const qPtr = w.allocOutput(m * n);
            const rPtr = w.allocOutput(n * n);
            w.exports.qrDecomp(wPtr, qPtr, rPtr, m, n);
            const qFlat = w.readF64(qPtr, m * n);
            const rFlat = w.readF64(rPtr, n * n);
            w.reset();
            const Q = A.like(m, n), R = A.like(n, n);
            for (let i=0;i<m*n;i++) Q.data[i] = A.zero.fromNumber(qFlat[i]);
            for (let i=0;i<n*n;i++) R.data[i] = A.zero.fromNumber(rFlat[i]);
            return { Q, R };
        }
    }

    const Q = A.like(m, n), R = A.like(n, n), W = A.clone();
    const qd = Q.data, rd = R.data, wd = W.data;
    for (let k=0;k<n;k++) {
        let normSq = A.zero;
        for (let i=0;i<m;i++) { const v=wd[i*n+k]; normSq=normSq.add(v.conjugate().multiply(v)); }
        const normK = normSq.sqrt();
        if (A.isZero(normK)) throw new Error(`qr: colonne linearmente dipendenti (colonna ${k}).`);
        rd[k*n+k] = normK;
        for (let i=0;i<m;i++) qd[i*n+k] = wd[i*n+k].divide(normK);
        for (let j=k+1;j<n;j++) {
            let dot = A.zero;
            for (let i=0;i<m;i++) dot = dot.add(qd[i*n+k].conjugate().multiply(wd[i*n+j]));
            rd[k*n+j] = dot;
            for (let i=0;i<m;i++) wd[i*n+j] = wd[i*n+j].subtract(dot.multiply(qd[i*n+k]));
        }
    }
    return { Q, R };
}
