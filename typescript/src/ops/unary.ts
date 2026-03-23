// ops/unary.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

type UnaryFn = "unaryAbs"|"unaryNeg"|"unarySqrt"|"unaryRound"|"unaryFloor"|"unaryCeil"|"unaryExp"|"unarySin"|"unaryCos"|"unaryTan";

function _applyUnary<T extends INumeric<T>>(
    A: Matrix<T>, fn: UnaryFn, opF64: (v: number) => number, opGen: (v: T) => T
): Matrix<T> {
    const d = A.data, len = d.length;
    const outData = new Array<T>(len);
    if (A.isFloat64) {
        if (len >= WASM_THRESHOLD.ELEMENTWISE) {
            const w = getBridgeSync();
            if (w) {
                const aPtr = w.writeFloat64M(d as any);
                const cPtr = w.allocOutput(len);
                (w.exports[fn] as any)(aPtr, cPtr, len);
                const flat = w.readF64(cPtr, len);
                w.reset();
                for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber(flat[i]);
                return A.likeWithData(A.rows, A.cols, outData);
            }
        }
        for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber(opF64((d[i] as any).value));
    } else {
        for (let i = 0; i < len; i++) outData[i] = opGen(d[i]);
    }
    return A.likeWithData(A.rows, A.cols, outData);
}

export const abs   = function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unaryAbs",   Math.abs,   x => x.abs()); };
export const sqrt  = function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unarySqrt",  Math.sqrt,  x => x.sqrt()); };
export const round = function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unaryRound", Math.round, x => x.round()); };
export const negate= function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unaryNeg",   v => -v,    x => x.negate()); };
export const exp   = function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unaryExp",   Math.exp,   x => x.fromNumber(Math.exp(x.toNumber()))); };
export const floor = function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unaryFloor", Math.floor, x => x.fromNumber(Math.floor(x.toNumber()))); };
export const ceil  = function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unaryCeil",  Math.ceil,  x => x.fromNumber(Math.ceil(x.toNumber()))); };
export const sin   = function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unarySin",   Math.sin,   x => x.fromNumber(Math.sin(x.toNumber()))); };
export const cos   = function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unaryCos",   Math.cos,   x => x.fromNumber(Math.cos(x.toNumber()))); };
export const tan   = function<T extends INumeric<T>>(this: Matrix<T>) { return _applyUnary(this, "unaryTan",   Math.tan,   x => x.fromNumber(Math.tan(x.toNumber()))); };

export function trace<T extends INumeric<T>>(this: Matrix<T>): T {
    if (this.isFloat64 && this.rows === this.cols && this.rows * this.rows >= WASM_THRESHOLD.STATS) {
        const w = getBridgeSync();
        if (w) {
            const aPtr = w.writeFloat64M(this.data as any);
            const r = w.exports.trace(aPtr, this.rows);
            w.reset();
            return this.zero.fromNumber(r);
        }
    }
    const n = Math.min(this.rows, this.cols), d = this.data, C = this.cols;
    let t = this.zero;
    for (let i = 0; i < n; i++) t = t.add(d[i * C + i]);
    return t;
}
