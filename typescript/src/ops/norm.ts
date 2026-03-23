// ops/norm.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

type NormType = "1"|"2"|"inf"|"fro";

export function norm<T extends INumeric<T>>(this: Matrix<T>, type: NormType = "2"): number {
    const isVec = this.rows === 1 || this.cols === 1;
    const len = this.data.length;
    if (this.isFloat64 && len >= WASM_THRESHOLD.STATS) {
        const w = getBridgeSync();
        if (w) {
            const aPtr = w.writeFloat64M(this.data as any);
            const R = this.rows, C = this.cols;
            let result: number;
            switch (type.toUpperCase()) {
                case "FRO": case "2":
                    result = w.exports.normFro(aPtr, len); break;
                case "1":
                    result = isVec ? w.exports.normVec1(aPtr, len) : w.exports.normMat1(aPtr, R, C); break;
                case "INF":
                    result = isVec ? w.exports.normVecInf(aPtr, len) : w.exports.normMatInf(aPtr, R, C); break;
                default:
                    w.reset();
                    throw new Error(`norm: tipo '${type}' non supportato.`);
            }
            w.reset();
            return result;
        }
    }
    return isVec ? _normVec(this, type) : _normMat(this, type);
}

function _normVec<T extends INumeric<T>>(A: Matrix<T>, type: NormType): number {
    const d = A.data, len = d.length, f64 = A.isFloat64;
    switch (type.toUpperCase()) {
        case "1": {
            let s = 0;
            if (f64) for (let i=0;i<len;i++){const v=(d[i] as any).value;s+=v<0?-v:v;}
            else for (let i=0;i<len;i++) s+=d[i].abs().toNumber();
            return s;
        }
        case "INF": {
            let m = 0;
            if (f64) for (let i=0;i<len;i++){const v=Math.abs((d[i] as any).value);if(v>m)m=v;}
            else for (let i=0;i<len;i++){const v=d[i].abs().toNumber();if(v>m)m=v;}
            return m;
        }
        default: {
            let ss = 0;
            if (f64) for (let i=0;i<len;i++){const v=(d[i] as any).value;ss+=v*v;}
            else for (let i=0;i<len;i++){const v=d[i].abs().toNumber();ss+=v*v;}
            return Math.sqrt(ss);
        }
    }
}

function _normMat<T extends INumeric<T>>(A: Matrix<T>, type: NormType): number {
    const R = A.rows, C = A.cols, d = A.data, f64 = A.isFloat64;
    switch (type.toUpperCase()) {
        case "1": {
            let max = 0;
            for (let j=0;j<C;j++){let s=0;if(f64)for(let i=0;i<R;i++){const v=(d[i*C+j] as any).value;s+=v<0?-v:v;}else for(let i=0;i<R;i++)s+=A.get(i,j).abs().toNumber();if(s>max)max=s;}
            return max;
        }
        case "INF": {
            let max = 0;
            for (let i=0;i<R;i++){const off=i*C;let s=0;if(f64)for(let j=0;j<C;j++){const v=(d[off+j] as any).value;s+=v<0?-v:v;}else for(let j=0;j<C;j++)s+=A.get(i,j).abs().toNumber();if(s>max)max=s;}
            return max;
        }
        case "FRO": {
            let ss = 0;
            if (f64) for (let i=0;i<R*C;i++){const v=(d[i] as any).value;ss+=v*v;}
            else for (let i=0;i<R*C;i++){const v=d[i].abs().toNumber();ss+=v*v;}
            return Math.sqrt(ss);
        }
        default: throw new Error(`norm: tipo '${type}' non supportato.`);
    }
}
