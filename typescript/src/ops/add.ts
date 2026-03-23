// ops/add.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

export function add<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") return _addScalar(this, B);
    if (this.rows === B.rows && this.cols === B.cols) return _addMatrix(this, B);
    if (B.rows === 1 && B.cols === this.cols) return _addRowVec(this, B);
    if (B.cols === 1 && B.rows === this.rows) return _addColVec(this, B);
    throw new Error(`add: dimensioni incompatibili ${this.rows}×${this.cols} e ${B.rows}×${B.cols}`);
}

export function totalSum<T extends INumeric<T>>(this: Matrix<T>): T {
    if (this.isFloat64 && this.data.length >= WASM_THRESHOLD.STATS) {
        const w = getBridgeSync();
        if (w) {
            const ptr = w.writeFloat64M(this.data as any);
            const s   = w.exports.totalSum(ptr, this.data.length);
            w.reset();
            return this.zero.fromNumber(s);
        }
    }
    const d = this.data, len = d.length;
    let s = this.zero;
    for (let i = 0; i < len; i++) s = s.add(d[i]);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────

function _addScalar<T extends INumeric<T>>(A: Matrix<T>, scalar: number): Matrix<T> {
    const d = A.data, len = d.length;
    const outData = new Array<T>(len);
    if (A.isFloat64) {
        if (len >= WASM_THRESHOLD.ELEMENTWISE) {
            const w = getBridgeSync();
            if (w) {
                const aPtr = w.writeFloat64M(d as any);
                const cPtr = w.allocOutput(len);
                w.exports.addScalar(aPtr, cPtr, len, scalar);
                const flat = w.readF64(cPtr, len);
                w.reset();
                for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber(flat[i]);
                return A.likeWithData(A.rows, A.cols, outData);
            }
        }
        for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber((d[i] as any).value + scalar);
    } else {
        const s = A.zero.fromNumber(scalar);
        for (let i = 0; i < len; i++) outData[i] = d[i].add(s);
    }
    return A.likeWithData(A.rows, A.cols, outData);
}

function _addMatrix<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const ad = A.data, bd = B.data, len = ad.length;
    const outData = new Array<T>(len);
    if (A.isFloat64 && (B as any).isFloat64) {
        if (len >= WASM_THRESHOLD.ELEMENTWISE) {
            const w = getBridgeSync();
            if (w) {
                const aPtr = w.writeFloat64M(ad as any);
                const bPtr = w.writeFloat64M(bd as any);
                const cPtr = w.allocOutput(len);
                w.exports.addMatrix(aPtr, bPtr, cPtr, len);
                const flat = w.readF64(cPtr, len);
                w.reset();
                for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber(flat[i]);
                return A.likeWithData(A.rows, A.cols, outData);
            }
        }
        for (let i = 0; i < len; i++)
            outData[i] = A.zero.fromNumber((ad[i] as any).value + (bd[i] as any).value);
    } else {
        for (let i = 0; i < len; i++) outData[i] = ad[i].add(bd[i]);
    }
    return A.likeWithData(A.rows, A.cols, outData);
}

function _addRowVec<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const R = A.rows, C = A.cols;
    const ad = A.data, bd = B.data;
    const outData = new Array<T>(R * C);
    if (A.isFloat64 && (B as any).isFloat64) {
        if (R * C >= WASM_THRESHOLD.ELEMENTWISE) {
            const w = getBridgeSync();
            if (w) {
                const aPtr = w.writeFloat64M(ad as any);
                const bPtr = w.writeFloat64M(bd as any);
                const cPtr = w.allocOutput(R * C);
                w.exports.addRowVec(aPtr, bPtr, cPtr, R, C);
                const flat = w.readF64(cPtr, R * C);
                w.reset();
                for (let i = 0; i < R * C; i++) outData[i] = A.zero.fromNumber(flat[i]);
                return A.likeWithData(R, C, outData);
            }
        }
        for (let i = 0; i < R; i++) {
            const off = i * C;
            for (let j = 0; j < C; j++)
                outData[off + j] = A.zero.fromNumber((ad[off + j] as any).value + (bd[j] as any).value);
        }
    } else {
        for (let i = 0; i < R; i++) {
            const off = i * C;
            for (let j = 0; j < C; j++) outData[off + j] = ad[off + j].add(bd[j]);
        }
    }
    return A.likeWithData(R, C, outData);
}

function _addColVec<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const R = A.rows, C = A.cols;
    const ad = A.data, bd = B.data;
    const outData = new Array<T>(R * C);
    if (A.isFloat64 && (B as any).isFloat64) {
        if (R * C >= WASM_THRESHOLD.ELEMENTWISE) {
            const w = getBridgeSync();
            if (w) {
                const aPtr = w.writeFloat64M(ad as any);
                const bPtr = w.writeFloat64M(bd as any);
                const cPtr = w.allocOutput(R * C);
                w.exports.addColVec(aPtr, bPtr, cPtr, R, C);
                const flat = w.readF64(cPtr, R * C);
                w.reset();
                for (let i = 0; i < R * C; i++) outData[i] = A.zero.fromNumber(flat[i]);
                return A.likeWithData(R, C, outData);
            }
        }
        for (let i = 0; i < R; i++) {
            const off = i * C, bv = (bd[i] as any).value;
            for (let j = 0; j < C; j++)
                outData[off + j] = A.zero.fromNumber((ad[off + j] as any).value + bv);
        }
    } else {
        for (let i = 0; i < R; i++) {
            const off = i * C, bv = bd[i];
            for (let j = 0; j < C; j++) outData[off + j] = ad[off + j].add(bv);
        }
    }
    return A.likeWithData(R, C, outData);
}
