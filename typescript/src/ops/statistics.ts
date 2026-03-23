// ops/statistics.ts — con fast-path WASM (SIMD su sum/max/min)
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync } from "../wasm/wasm_bridge";

export function max<T extends INumeric<T>>(this: Matrix<T>, dim: 1|2 = 1): { value: Matrix<T>; index: Int32Array } {
    if (this.isFloat64) {
        const w = getBridgeSync();
        if (w) {
            const R = this.rows, C = this.cols;
            const aPtr   = w.writeFloat64M(this.data as any);
            const outPtr = w.allocOutput(dim === 1 ? C : R);
            const idxPtr = w.allocI32(dim === 1 ? C : R);
            if (dim === 1) w.exports.maxCols(aPtr, outPtr, idxPtr, R, C);
            else           w.exports.maxRows(aPtr, outPtr, idxPtr, R, C);
            const flat = w.readF64(outPtr, dim === 1 ? C : R);
            const idxArr = w.readI32Array(idxPtr, dim === 1 ? C : R);
            w.reset();
            const vOut = dim === 1 ? this.like(1, C) : this.like(R, 1);
            for (let i = 0; i < flat.length; i++) vOut.data[i] = this.zero.fromNumber(flat[i]);
            return { value: vOut, index: idxArr };
        }
    }
    return dim === 1 ? _colReduce(this, true) : _rowReduce(this, true);
}

export function min<T extends INumeric<T>>(this: Matrix<T>, dim: 1|2 = 1): { value: Matrix<T>; index: Int32Array } {
    if (this.isFloat64) {
        const w = getBridgeSync();
        if (w) {
            const R = this.rows, C = this.cols;
            const aPtr   = w.writeFloat64M(this.data as any);
            const outPtr = w.allocOutput(dim === 1 ? C : R);
            const idxPtr = w.allocI32(dim === 1 ? C : R);
            if (dim === 1) w.exports.minCols(aPtr, outPtr, idxPtr, R, C);
            else           w.exports.minRows(aPtr, outPtr, idxPtr, R, C);
            const flat   = w.readF64(outPtr, dim === 1 ? C : R);
            const idxArr = w.readI32Array(idxPtr, dim === 1 ? C : R);
            w.reset();
            const vOut = dim === 1 ? this.like(1, C) : this.like(R, 1);
            for (let i = 0; i < flat.length; i++) vOut.data[i] = this.zero.fromNumber(flat[i]);
            return { value: vOut, index: idxArr };
        }
    }
    return dim === 1 ? _colReduce(this, false) : _rowReduce(this, false);
}

export function sum<T extends INumeric<T>>(this: Matrix<T>, dim: 1|2 = 1): Matrix<T> {
    if (this.isFloat64) {
        const w = getBridgeSync();
        if (w) {
            const R = this.rows, C = this.cols;
            const aPtr   = w.writeFloat64M(this.data as any);
            const outLen = dim === 1 ? C : R;
            const outPtr = w.allocOutput(outLen);
            if (dim === 1) w.exports.sumCols(aPtr, outPtr, R, C);
            else           w.exports.sumRows(aPtr, outPtr, R, C);
            const flat = w.readF64(outPtr, outLen);
            w.reset();
            const vOut = dim === 1 ? this.like(1, C) : this.like(R, 1);
            for (let i = 0; i < outLen; i++) vOut.data[i] = this.zero.fromNumber(flat[i]);
            return vOut;
        }
    }
    return dim === 1 ? _sumCols(this) : _sumRows(this);
}

export function mean<T extends INumeric<T>>(this: Matrix<T>, dim: 1|2 = 1): Matrix<T> {
    const s   = (sum<T>).call(this, dim);
    const div = this.zero.fromNumber(dim === 1 ? this.rows : this.cols);
    const d = s.data, len = d.length;
    for (let i = 0; i < len; i++) d[i] = d[i].divide(div);
    return s;
}

// ── TS fallback ────────────────────────────────────────────────────────────

function _colReduce<T extends INumeric<T>>(A: Matrix<T>, wantMax: boolean): { value: Matrix<T>; index: Int32Array } {
    const R = A.rows, C = A.cols;
    const vOut = A.like(1, C);
    const iOut = new Int32Array(C);
    const ad = A.data, od = vOut.data;
    for (let j = 0; j < C; j++) { od[j] = ad[j]; iOut[j] = 1; }
    for (let i = 1; i < R; i++) {
        const off = i * C;
        for (let j = 0; j < C; j++) {
            const v = ad[off + j];
            if (wantMax ? v.greaterThan(od[j]) : v.lessThan(od[j])) { od[j] = v; iOut[j] = i + 1; }
        }
    }
    return { value: vOut, index: iOut };
}

function _rowReduce<T extends INumeric<T>>(A: Matrix<T>, wantMax: boolean): { value: Matrix<T>; index: Int32Array } {
    const R = A.rows, C = A.cols;
    const vOut = A.like(R, 1);
    const iOut = new Int32Array(R);
    const ad = A.data, od = vOut.data;
    for (let i = 0; i < R; i++) {
        const off = i * C;
        let best = ad[off], bestJ = 1;
        for (let j = 1; j < C; j++) {
            const v = ad[off + j];
            if (wantMax ? v.greaterThan(best) : v.lessThan(best)) { best = v; bestJ = j + 1; }
        }
        od[i] = best; iOut[i] = bestJ;
    }
    return { value: vOut, index: iOut };
}

function _sumCols<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    const R = A.rows, C = A.cols;
    const out = A.like(1, C);
    const ad = A.data, od = out.data;
    if (A.isFloat64) {
        const acc = new Float64Array(C);
        for (let i = 0; i < R; i++) { const off = i*C; for (let j = 0; j < C; j++) acc[j] += (ad[off+j] as any).value; }
        for (let j = 0; j < C; j++) od[j] = A.zero.fromNumber(acc[j]);
    } else {
        for (let i = 0; i < R; i++) { const off = i*C; for (let j = 0; j < C; j++) od[j] = od[j].add(ad[off+j]); }
    }
    return out;
}

function _sumRows<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    const R = A.rows, C = A.cols;
    const out = A.like(R, 1);
    const ad = A.data, od = out.data;
    if (A.isFloat64) {
        for (let i = 0; i < R; i++) {
            const off = i*C; let s = 0;
            for (let j = 0; j < C; j++) s += (ad[off+j] as any).value;
            od[i] = A.zero.fromNumber(s);
        }
    } else {
        for (let i = 0; i < R; i++) {
            const off = i*C; let s = A.zero;
            for (let j = 0; j < C; j++) s = s.add(ad[off+j]);
            od[i] = s;
        }
    }
    return out;
}
