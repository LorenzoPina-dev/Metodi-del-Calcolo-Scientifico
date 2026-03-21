// ops/statistics.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function max<T extends INumeric<T>>(this: Matrix<T>, dim: 1 | 2 = 1): { value: Matrix<T>; index: Int32Array } {
    return dim === 1 ? _colReduce(this, true) : _rowReduce(this, true);
}

export function min<T extends INumeric<T>>(this: Matrix<T>, dim: 1 | 2 = 1): { value: Matrix<T>; index: Int32Array } {
    return dim === 1 ? _colReduce(this, false) : _rowReduce(this, false);
}

export function sum<T extends INumeric<T>>(this: Matrix<T>, dim: 1 | 2 = 1): Matrix<T> {
    return dim === 1 ? _sumCols(this) : _sumRows(this);
}

export function mean<T extends INumeric<T>>(this: Matrix<T>, dim: 1 | 2 = 1): Matrix<T> {
    const s = sum.call(this, dim);
    const div = this.zero.fromNumber(dim === 1 ? this.rows : this.cols);
    const d = s.data, len = d.length;
    for (let i = 0; i < len; i++) d[i] = d[i].divide(div);
    return s;
}

// ==================== PRIVATE ====================

function _colReduce<T extends INumeric<T>>(
    A: Matrix<T>, wantMax: boolean
): { value: Matrix<T>; index: Int32Array } {
    const R = A.rows, C = A.cols;
    const vOut = A.like(1, C);
    const iOut = new Int32Array(C);
    const ad = A.data, od = vOut.data;

    // Inizializza con la riga 0
    for (let j = 0; j < C; j++) { od[j] = ad[j]; iOut[j] = 1; }

    for (let i = 1; i < R; i++) {
        const off = i * C;
        for (let j = 0; j < C; j++) {
            const v = ad[off + j];
            const better = wantMax ? v.greaterThan(od[j]) : v.lessThan(od[j]);
            if (better) { od[j] = v; iOut[j] = i + 1; }
        }
    }
    return { value: vOut, index: iOut };
}

function _rowReduce<T extends INumeric<T>>(
    A: Matrix<T>, wantMax: boolean
): { value: Matrix<T>; index: Int32Array } {
    const R = A.rows, C = A.cols;
    const vOut = A.like(R, 1);
    const iOut = new Int32Array(R);
    const ad = A.data, od = vOut.data;

    for (let i = 0; i < R; i++) {
        const off = i * C;
        let best = ad[off], bestJ = 1;
        for (let j = 1; j < C; j++) {
            const v = ad[off + j];
            const better = wantMax ? v.greaterThan(best) : v.lessThan(best);
            if (better) { best = v; bestJ = j + 1; }
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
        // Accumula direttamente i valori float senza allocare Float64M intermedi
        const acc = new Float64Array(C);
        for (let i = 0; i < R; i++) {
            const off = i * C;
            let j = 0;
            for (; j <= C - 4; j += 4) {
                acc[j]     += (ad[off + j]     as any).value;
                acc[j + 1] += (ad[off + j + 1] as any).value;
                acc[j + 2] += (ad[off + j + 2] as any).value;
                acc[j + 3] += (ad[off + j + 3] as any).value;
            }
            for (; j < C; j++) acc[j] += (ad[off + j] as any).value;
        }
        for (let j = 0; j < C; j++) od[j] = A.zero.fromNumber(acc[j]);
    } else {
        for (let i = 0; i < R; i++) {
            const off = i * C;
            for (let j = 0; j < C; j++) od[j] = od[j].add(ad[off + j]);
        }
    }
    return out;
}

function _sumRows<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    const R = A.rows, C = A.cols;
    const out = A.like(R, 1);
    const ad = A.data, od = out.data;

    if (A.isFloat64) {
        for (let i = 0; i < R; i++) {
            const off = i * C;
            let s = 0;
            let j = 0;
            for (; j <= C - 4; j += 4) {
                s += (ad[off + j] as any).value + (ad[off + j + 1] as any).value
                   + (ad[off + j + 2] as any).value + (ad[off + j + 3] as any).value;
            }
            for (; j < C; j++) s += (ad[off + j] as any).value;
            od[i] = A.zero.fromNumber(s);
        }
    } else {
        for (let i = 0; i < R; i++) {
            const off = i * C;
            let s = A.zero;
            for (let j = 0; j < C; j++) s = s.add(ad[off + j]);
            od[i] = s;
        }
    }
    return out;
}
