// ops/multiply.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

export function multiply<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") {
        if (this.isFloat64) {
            const len = this.data.length;
            if (len >= WASM_THRESHOLD.MATMUL) {
                const w = getBridgeSync();
                if (w) {
                    const aPtr = w.writeFloat64M(this.data as any);
                    const cPtr = w.allocOutput(len);
                    w.exports.mulScalar(aPtr, cPtr, len, B);
                    const flat = w.readF64(cPtr, len);
                    w.reset();
                    const out = new Array<T>(len);
                    for (let i = 0; i < len; i++) out[i] = this.zero.fromNumber(flat[i]);
                    return this.likeWithData(this.rows, this.cols, out);
                }
            }
            const d = this.data, outData = new Array<T>(len);
            for (let i = 0; i < len; i++) outData[i] = this.zero.fromNumber((d[i] as any).value * B);
            return this.likeWithData(this.rows, this.cols, outData);
        }
        const s = this.zero.fromNumber(B), d = this.data, len = d.length;
        const outData = new Array<T>(len);
        for (let i = 0; i < len; i++) outData[i] = d[i].multiply(s);
        return this.likeWithData(this.rows, this.cols, outData);
    }

    if (this.cols !== B.rows)
        throw new Error(`multiply: dimensioni interne non coincidono (${this.cols} ≠ ${B.rows})`);

    if (this.isFloat64 && (B as any).isFloat64) {
        const M = this.rows, K = this.cols, N = B.cols;
        // Usa WASM solo se il problema è abbastanza grande da ammortizzare l'overhead
        if (M * K >= WASM_THRESHOLD.MATMUL) {
            const w = getBridgeSync();
            if (w) {
                const aPtr = w.writeFloat64M(this.data as any);
                const bPtr = w.writeFloat64M(B.data as any);
                const cPtr = w.allocOutput(M * N);
                w.exports.matmul(aPtr, bPtr, cPtr, M, K, N);
                const flat = w.readF64(cPtr, M * N);
                w.reset();
                const outData = new Array<T>(M * N);
                for (let i = 0; i < M * N; i++) outData[i] = this.zero.fromNumber(flat[i]);
                return this.likeWithData(M, N, outData);
            }
        }
        return _matMulF64(this, B as any) as any;
    }
    return _matMulGeneric(this, B);
}

function _matMulF64<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const M = A.rows, K = A.cols, N = B.cols;
    const af = A.data, bf = B.data;
    const cf = new Float64Array(M * N);
    for (let i = 0; i < M; i++) {
        const iOff = i * K, outOff = i * N;
        for (let k = 0; k < K; k++) {
            const aik = (af[iOff + k] as any).value as number;
            if (aik === 0) continue;
            const kOff = k * N;
            for (let j = 0; j < N; j++) cf[outOff + j] += aik * (bf[kOff + j] as any).value;
        }
    }
    const outData = new Array<T>(M * N);
    for (let i = 0; i < M * N; i++) outData[i] = A.zero.fromNumber(cf[i]);
    return A.likeWithData(M, N, outData);
}

function _matMulGeneric<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const M = A.rows, K = A.cols, N = B.cols;
    const out = A.like(M, N);
    const ad = A.data, bd = B.data, od = out.data;
    for (let i = 0; i < M; i++) {
        const iOff = i * K, outOff = i * N;
        for (let k = 0; k < K; k++) {
            const aik = ad[iOff + k], kOff = k * N;
            for (let j = 0; j < N; j++)
                od[outOff + j] = od[outOff + j].add(aik.multiply(bd[kOff + j]));
        }
    }
    return out;
}
