// decomposition/lup.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

export function lup<T extends INumeric<T>>(
    A: Matrix<T>
): { L: Matrix<T>; U: Matrix<T>; P: number[]; swaps: number } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("lup: matrice non quadrata.");

    if (A.isFloat64 && n * n >= WASM_THRESHOLD.DECOMP) {
        const w = getBridgeSync();
        if (w) {
            const wPtr = w.writeFloat64M(A.data as any);
            const pPtr = w.allocI32(n);
            const swaps = w.exports.lupDecomp(wPtr, pPtr, n);
            if (swaps < 0) throw new Error("lup: matrice singolare.");
            const wFlat = w.readF64(wPtr, n * n);
            const P = w.readPermutation(pPtr, n);
            w.reset();
            const L = A.likeIdentity(n), U = A.like(n, n);
            const ld = L.data, ud = U.data;
            for (let i=0;i<n;i++) { const off=i*n; for (let j=0;j<n;j++) { if(i>j)ld[off+j]=A.zero.fromNumber(wFlat[off+j]); else ud[off+j]=A.zero.fromNumber(wFlat[off+j]); } }
            return { L, U, P, swaps };
        }
    }

    const W = A.clone(), wd = W.data;
    const P = Array.from({length: n}, (_,i) => i);
    let swaps = 0;
    for (let i=0;i<n;i++) {
        let maxVal = wd[i*n+i].abs(), maxRow = i;
        for (let k=i+1;k<n;k++) { const v=wd[k*n+i].abs(); if(v.greaterThan(maxVal)){maxVal=v;maxRow=k;} }
        if (W.isZero(maxVal)) throw new Error("lup: matrice singolare.");
        if (maxRow!==i) { swaps++; const iO=i*n,mO=maxRow*n; for(let j=0;j<n;j++){const t=wd[iO+j];wd[iO+j]=wd[mO+j];wd[mO+j]=t;} [P[i],P[maxRow]]=[P[maxRow],P[i]]; }
        const pivot = wd[i*n+i];
        for (let j=i+1;j<n;j++) { const jO=j*n; const f=wd[jO+i].divide(pivot); wd[jO+i]=f; for(let k=i+1;k<n;k++) wd[jO+k]=wd[jO+k].subtract(f.multiply(wd[i*n+k])); }
    }
    const L=A.likeIdentity(n), U=A.like(n,n); const ld=L.data,ud=U.data;
    for (let i=0;i<n;i++) { const off=i*n; for(let j=0;j<n;j++){if(i>j)ld[off+j]=wd[off+j];else ud[off+j]=wd[off+j];} }
    return { L, U, P, swaps };
}
