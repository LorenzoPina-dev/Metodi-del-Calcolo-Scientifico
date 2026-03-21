// ops/statistics.ts
import { Matrix } from "..";
import { INumeric } from "../type";

// ---- MAX ----
export function max<T extends INumeric<T>>(
    this: Matrix<T>,
    dim: 1 | 2 = 1
): { value: Matrix<T>; index: Int32Array } {
    return dim === 1
        ? columnReduction(this, (a, b) => a.greaterThan(b))
        : rowReduction(this, (a, b) => a.greaterThan(b));
}

// ---- MIN ----
export function min<T extends INumeric<T>>(
    this: Matrix<T>,
    dim: 1 | 2 = 1
): { value: Matrix<T>; index: Int32Array } {
    return dim === 1
        ? columnReduction(this, (a, b) => a.lessThan(b))
        : rowReduction(this, (a, b) => a.lessThan(b));
}

// ---- SUM ----
export function sum<T extends INumeric<T>>(this: Matrix<T>, dim: 1 | 2 = 1): Matrix<T> {
    return dim === 1 ? sumColumns(this) : sumRows(this);
}

// ---- MEAN ----
export function mean<T extends INumeric<T>>(this: Matrix<T>, dim: 1 | 2 = 1): Matrix<T> {
    const s = sum.call(this, dim);
    const divisor = this.zero.fromNumber(dim === 1 ? this.rows : this.cols);
    for (let i = 0; i < s.data.length; i++) s.data[i] = s.data[i].divide(divisor);
    return s;
}

// ==================== PRIVATI ====================

function columnReduction<T extends INumeric<T>>(
    A: Matrix<T>,
    isBetter: (candidate: T, current: T) => boolean
): { value: Matrix<T>; index: Int32Array } {
    const { rows: R, cols: C } = A;
    const vOut = A.like(1, C);
    const iOut = new Int32Array(C);

    // Inizializza con il primo elemento di ogni colonna
    for (let j = 0; j < C; j++) {
        vOut.data[j] = A.data[j]; // riga 0
        iOut[j] = 1;
    }

    for (let i = 1; i < R; i++) {
        const off = i * C;
        for (let j = 0; j < C; j++) {
            if (isBetter(A.data[off + j], vOut.data[j])) {
                vOut.data[j] = A.data[off + j];
                iOut[j] = i + 1; // 1-based
            }
        }
    }
    return { value: vOut, index: iOut };
}

function rowReduction<T extends INumeric<T>>(
    A: Matrix<T>,
    isBetter: (candidate: T, current: T) => boolean
): { value: Matrix<T>; index: Int32Array } {
    const { rows: R, cols: C } = A;
    const vOut = A.like(R, 1);
    const iOut = new Int32Array(R);

    for (let i = 0; i < R; i++) {
        const off = i * C;
        let best = A.data[off];
        let bestJ = 1;
        for (let j = 1; j < C; j++) {
            if (isBetter(A.data[off + j], best)) {
                best = A.data[off + j];
                bestJ = j + 1; // 1-based
            }
        }
        vOut.data[i] = best;
        iOut[i] = bestJ;
    }
    return { value: vOut, index: iOut };
}

function sumColumns<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    const out = A.like(1, A.cols);
    const { rows, cols } = A;
    for (let i = 0; i < rows; i++) {
        const off = i * cols;
        for (let j = 0; j < cols; j++) {
            out.data[j] = out.data[j].add(A.data[off + j]);
        }
    }
    return out;
}

function sumRows<T extends INumeric<T>>(A: Matrix<T>): Matrix<T> {
    const out = A.like(A.rows, 1);
    const { rows, cols } = A;
    for (let i = 0; i < rows; i++) {
        const off = i * cols;
        let s = A.zero;
        for (let j = 0; j < cols; j++) s = s.add(A.data[off + j]);
        out.data[i] = s;
    }
    return out;
}
