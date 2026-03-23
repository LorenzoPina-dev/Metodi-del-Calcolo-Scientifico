// ops/dotOps.ts — con fast-path WASM (SIMD su dotMul/dotDiv/dotPow broadcast)
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync } from "../wasm/wasm_bridge";

export function dotMultiply<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") return _applyScalar(this, B, "mulScalar", (a,b) => a*b, (a,b) => a.multiply(b));
    return _applyBroadcast(this, B, "dotMul", "dotMulRowVec", "dotMulColVec",
        (a,b) => a*b, (a,b) => a*b, (a,b) => a*b,
        (a,b) => a.multiply(b));
}

export function dotDivide<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") return _applyScalar(this, B, "mulScalar", (a,b) => a/b, (a,b) => a.divide(b), true);
    return _applyBroadcast(this, B, "dotDiv", null, null,
        (a,b) => a/b, (a,b) => a/b, (a,b) => a/b,
        (a,b) => a.divide(b));
}

export function dotPow<T extends INumeric<T>>(this: Matrix<T>, exp: number | Matrix<T>): Matrix<T> {
    if (typeof exp === "number") {
        if (this.isFloat64) {
            const w = getBridgeSync();
            if (w) {
                const d = this.data, len = d.length;
                const aPtr = w.writeFloat64M(d as any);
                const cPtr = w.allocOutput(len);
                w.exports.dotPowScalar(aPtr, cPtr, len, exp);
                const flat = w.readF64(cPtr, len);
                w.reset();
                const outData = new Array<T>(len);
                for (let i = 0; i < len; i++) outData[i] = this.zero.fromNumber(flat[i]);
                return this.likeWithData(this.rows, this.cols, outData);
            }
            // TS F64 fallback
            const d = this.data, len = d.length;
            const outData = new Array<T>(len);
            for (let i = 0; i < len; i++) outData[i] = this.zero.fromNumber(Math.pow((d[i] as any).value, exp));
            return this.likeWithData(this.rows, this.cols, outData);
        }
        return _applyScalarGeneric(this, a => _powGen(a, exp));
    }
    return _applyBroadcast(this, exp, null, null, null,
        (a,b) => Math.pow(a,b), (a,b) => Math.pow(a,b), (a,b) => Math.pow(a,b),
        (a,b) => _powGen(a, b.toNumber()));
}

// ─── helpers ─────────────────────────────────────────────────────────────────

type WasmBinFn = "dotMul" | "dotDiv" | "addMatrix" | "subMatrix";
type WasmBcastFn = "dotMulRowVec" | "dotMulColVec" | "addRowVec" | "subRowVec" | "addColVec" | "subColVec" | null;

function _applyScalar<T extends INumeric<T>>(
    A: Matrix<T>, s: number,
    wasmFn: string,
    opF64: (a: number, b: number) => number,
    opGen: (a: T, b: T) => T,
    invertScalar = false
): Matrix<T> {
    const d = A.data, len = d.length;
    const outData = new Array<T>(len);
    if (A.isFloat64) {
        const w = getBridgeSync();
        if (w) {
            const scalar = invertScalar ? 1 / s : s;
            const aPtr = w.writeFloat64M(d as any);
            const cPtr = w.allocOutput(len);
            (w.exports as any)[wasmFn](aPtr, cPtr, len, scalar);
            const flat = w.readF64(cPtr, len);
            w.reset();
            for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber(flat[i]);
            return A.likeWithData(A.rows, A.cols, outData);
        }
        for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber(opF64((d[i] as any).value, s));
    } else {
        const sv = A.zero.fromNumber(s);
        for (let i = 0; i < len; i++) outData[i] = opGen(d[i], sv);
    }
    return A.likeWithData(A.rows, A.cols, outData);
}

function _applyScalarGeneric<T extends INumeric<T>>(A: Matrix<T>, op: (a: T) => T): Matrix<T> {
    const d = A.data, len = d.length;
    const outData = new Array<T>(len);
    for (let i = 0; i < len; i++) outData[i] = op(d[i]);
    return A.likeWithData(A.rows, A.cols, outData);
}

function _applyBroadcast<T extends INumeric<T>>(
    A: Matrix<T>, B: Matrix<T>,
    wasmFull: WasmBinFn | null,
    wasmRow:  WasmBcastFn,
    wasmCol:  WasmBcastFn,
    opF64Full: (a: number, b: number) => number,
    opF64Row:  (a: number, b: number) => number,
    opF64Col:  (a: number, b: number) => number,
    opGen: (a: T, b: T) => T
): Matrix<T> {
    const R = A.rows, C = A.cols;
    const ad = A.data, bd = B.data;
    const f64 = A.isFloat64 && (B as any).isFloat64;
    const outData = new Array<T>(R * C);

    const sameSize  = A.rows === B.rows && A.cols === B.cols;
    const isRowVec  = B.rows === 1 && B.cols === C;
    const isColVec  = B.cols === 1 && B.rows === R;

    if (!sameSize && !isRowVec && !isColVec)
        throw new Error("dotOp: dimensioni incompatibili");

    if (f64) {
        const w = getBridgeSync();
        if (w) {
            const aPtr = w.writeFloat64M(ad as any);
            const bPtr = w.writeFloat64M(bd as any);
            const cPtr = w.allocOutput(R * C);
            if (sameSize && wasmFull)
                (w.exports as any)[wasmFull](aPtr, bPtr, cPtr, R * C);
            else if (isRowVec && wasmRow)
                (w.exports as any)[wasmRow](aPtr, bPtr, cPtr, R, C);
            else if (isColVec && wasmCol)
                (w.exports as any)[wasmCol](aPtr, bPtr, cPtr, R, C);
            else {
                // fallback scalare in TS
                w.reset();
                const opScalar = sameSize ? opF64Full : isRowVec ? opF64Row : opF64Col;
                for (let i = 0; i < R; i++) {
                    const off = i * C;
                    for (let j = 0; j < C; j++) {
                        const bIdx = sameSize ? off+j : isRowVec ? j : i;
                        outData[off+j] = A.zero.fromNumber(opScalar((ad[off+j] as any).value, (bd[bIdx] as any).value));
                    }
                }
                return A.likeWithData(R, C, outData);
            }
            const flat = w.readF64(cPtr, R * C);
            w.reset();
            for (let i = 0; i < R*C; i++) outData[i] = A.zero.fromNumber(flat[i]);
            return A.likeWithData(R, C, outData);
        }
        // TS F64 fallback
        if (sameSize) {
            for (let i = 0; i < R*C; i++) outData[i] = A.zero.fromNumber(opF64Full((ad[i] as any).value, (bd[i] as any).value));
        } else if (isRowVec) {
            for (let i = 0; i < R; i++) { const off = i*C; for (let j = 0; j < C; j++) outData[off+j] = A.zero.fromNumber(opF64Row((ad[off+j] as any).value, (bd[j] as any).value)); }
        } else {
            for (let i = 0; i < R; i++) { const off = i*C, bv = (bd[i] as any).value; for (let j = 0; j < C; j++) outData[off+j] = A.zero.fromNumber(opF64Col((ad[off+j] as any).value, bv)); }
        }
    } else {
        if (sameSize) {
            for (let i = 0; i < R*C; i++) outData[i] = opGen(ad[i], bd[i]);
        } else if (isRowVec) {
            for (let i = 0; i < R; i++) { const off = i*C; for (let j = 0; j < C; j++) outData[off+j] = opGen(ad[off+j], bd[j]); }
        } else {
            for (let i = 0; i < R; i++) { const off = i*C, bv = bd[i]; for (let j = 0; j < C; j++) outData[off+j] = opGen(ad[off+j], bv); }
        }
    }
    return A.likeWithData(R, C, outData);
}

function _powGen<T extends INumeric<T>>(base: T, exp: number): T {
    if (!Number.isInteger(exp)) return base.fromNumber(Math.pow(base.toNumber(), exp));
    if (exp === 0) return base.fromNumber(1);
    if (exp < 0)  return base.fromNumber(1).divide(_powGen(base, -exp));
    let result = base.fromNumber(1), b = base, e = exp;
    while (e > 0) {
        if (e & 1) result = result.multiply(b);
        b = b.multiply(b);
        e >>= 1;
    }
    return result;
}
