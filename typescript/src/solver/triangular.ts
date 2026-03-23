// solver/triangular.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

function _wasmTri<T extends INumeric<T>>(
    M: Matrix<T>, b: Matrix<T>,
    fn: "solveLower"|"solveLowerUnit"|"solveUpper"
): Matrix<T> | null {
    if (!M.isFloat64 || !(b as any).isFloat64) return null;
    const n = M.rows;
    if (n * n < WASM_THRESHOLD.TRIANGULAR) return null;
    const w = getBridgeSync();
    if (!w) return null;
    const mPtr = w.writeFloat64M(M.data as any);
    const bPtr = w.writeFloat64M(b.data as any);
    const xPtr = w.allocOutput(n * b.cols);
    w.exports[fn](mPtr, bPtr, xPtr, n, b.cols);
    const flat = w.readF64(xPtr, n * b.cols);
    w.reset();
    const out = M.like(n, b.cols) as Matrix<T>;
    for (let i=0;i<n*b.cols;i++) out.data[i] = M.zero.fromNumber(flat[i]);
    return out;
}

export function solveLowerTriangular<T extends INumeric<T>>(L: Matrix<T>, b: Matrix<T>): Matrix<T> {
    const N = L.rows;
    if (N !== L.cols) throw new Error("solveLowerTriangular: L deve essere quadrata.");
    const fast = _wasmTri(L, b, "solveLower");
    if (fast) return fast;
    const x = b.like(N, b.cols); const ld = L.data, bd = b.data, xd = x.data; const BC = b.cols;
    for (let col=0;col<BC;col++) {
        xd[col] = bd[col].divide(ld[0]);
        for (let i=1;i<N;i++) {
            let s = bd[i*BC+col];
            for (let k=0;k<i;k++) s = s.subtract(ld[i*N+k].multiply(xd[k*BC+col]));
            xd[i*BC+col] = s.divide(ld[i*N+i]);
        }
    }
    return x;
}

export function solveUpperTriangular<T extends INumeric<T>>(U: Matrix<T>, b: Matrix<T>): Matrix<T> {
    const N = U.rows;
    if (N !== U.cols) throw new Error("solveUpperTriangular: U deve essere quadrata.");
    const fast = _wasmTri(U, b, "solveUpper");
    if (fast) return fast;
    const x = b.like(N, b.cols); const ud = U.data, bd = b.data, xd = x.data; const BC = b.cols;
    for (let col=0;col<BC;col++) {
        xd[(N-1)*BC+col] = bd[(N-1)*BC+col].divide(ud[(N-1)*N+(N-1)]);
        for (let i=N-2;i>=0;i--) {
            let s = bd[i*BC+col];
            for (let k=i+1;k<N;k++) s = s.subtract(ud[i*N+k].multiply(xd[k*BC+col]));
            xd[i*BC+col] = s.divide(ud[i*N+i]);
        }
    }
    return x;
}
