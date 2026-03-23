// decomposition/cholesky.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

export function cholesky<T extends INumeric<T>>(A: Matrix<T>): { L: Matrix<T> } {
    if (A.rows !== A.cols) throw new Error("cholesky: matrice non quadrata.");
    if (!A.isSymmetric())  throw new Error("cholesky: matrice non simmetrica.");
    const N = A.rows;

    if (A.isFloat64 && N * N >= WASM_THRESHOLD.DECOMP) {
        const w = getBridgeSync();
        if (w) {
            const aPtr = w.writeFloat64M(A.data as any);
            const lPtr = w.allocOutput(N * N);
            const ret  = w.exports.choleskyDecomp(aPtr, lPtr, N);
            if (ret < 0) { w.reset(); throw new Error("cholesky: matrice non definita positiva."); }
            const lFlat = w.readF64(lPtr, N * N);
            w.reset();
            const L = A.like(N, N);
            for (let i=0;i<N*N;i++) L.data[i] = A.zero.fromNumber(lFlat[i]);
            return { L };
        }
    }

    const L = A.like(N, N), ad = A.data, ld = L.data;
    for (let j=0;j<N;j++) {
        let diagSum = A.zero;
        for (let k=0;k<j;k++) { const ljk=ld[j*N+k]; diagSum=diagSum.add(ljk.multiply(ljk)); }
        const d = ad[j*N+j].subtract(diagSum);
        if (d.negate().greaterThan(A.zero)) throw new Error("cholesky: matrice non definita positiva.");
        const ljj = d.sqrt();
        ld[j*N+j] = ljj;
        for (let i=j+1;i<N;i++) {
            let offSum = A.zero;
            for (let k=0;k<j;k++) offSum = offSum.add(ld[i*N+k].multiply(ld[j*N+k]));
            ld[i*N+j] = ad[i*N+j].subtract(offSum).divide(ljj);
        }
    }
    return { L };
}
