// ops/transform.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/** Trasposizione: A.' */
export function transpose<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    const out = this.like(this.cols, this.rows);
    const R = this.rows, C = this.cols;
    for (let i = 0; i < R; i++) {
        const iOff = i * C;
        for (let j = 0; j < C; j++) {
            out.data[j * R + i] = this.data[iOff + j];
        }
    }
    return out;
}

/** Reshape: cambia dimensioni mantenendo il numero di elementi. */
export function reshape<T extends INumeric<T>>(this: Matrix<T>, r: number, c: number): Matrix<T> {
    if (r * c !== this.rows * this.cols) {
        throw new Error("reshape: il numero totale di elementi deve rimanere invariato.");
    }
    const Ctor = this.constructor as new (r: number, c: number, z: T, o: T, d: Array<T>) => Matrix<T>;
    return new Ctor(r, c, this.zero, this.one, [...this.data]);
}

/** Flip: dim=1 verticale (flipud), dim=2 orizzontale (fliplr). */
export function flip<T extends INumeric<T>>(this: Matrix<T>, dim: 1 | 2 = 1): Matrix<T> {
    const out = this.like(this.rows, this.cols);
    const R = this.rows, C = this.cols;

    if (dim === 1) {
        for (let i = 0; i < R; i++) {
            const src = i * C;
            const dst = (R - 1 - i) * C;
            for (let j = 0; j < C; j++) out.data[dst + j] = this.data[src + j];
        }
    } else {
        for (let i = 0; i < R; i++) {
            const off = i * C;
            for (let j = 0; j < C; j++) {
                out.data[off + (C - 1 - j)] = this.data[off + j];
            }
        }
    }
    return out;
}

/** Rot90: ruota di 90° antiorario k volte. */
export function rot90<T extends INumeric<T>>(this: Matrix<T>, k: number = 1): Matrix<T> {
    k = ((k % 4) + 4) % 4;
    if (k === 0) return this.clone();

    if (k === 2) return flip.call(flip.call(this, 1), 2);

    const out = this.like(this.cols, this.rows);
    const R = this.rows, C = this.cols;

    if (k === 1) { // 90° antiorario
        for (let i = 0; i < R; i++)
            for (let j = 0; j < C; j++)
                out.data[(C - 1 - j) * R + i] = this.data[i * C + j];
    } else { // k === 3, 270° antiorario = 90° orario
        for (let i = 0; i < R; i++)
            for (let j = 0; j < C; j++)
                out.data[j * R + (R - 1 - i)] = this.data[i * C + j];
    }
    return out;
}

/** Circshift: scostamento circolare. */
export function circshift<T extends INumeric<T>>(this: Matrix<T>, rShift: number, cShift: number): Matrix<T> {
    const out = this.like(this.rows, this.cols);
    const R = this.rows, C = this.cols;
    rShift = ((rShift % R) + R) % R;
    cShift = ((cShift % C) + C) % C;
    for (let i = 0; i < R; i++) {
        const newI = (i + rShift) % R;
        for (let j = 0; j < C; j++) {
            out.data[newI * C + ((j + cShift) % C)] = this.data[i * C + j];
        }
    }
    return out;
}

/** Repmat: replica la matrice in un blocco r×c. */
export function repmat<T extends INumeric<T>>(this: Matrix<T>, r: number, c: number): Matrix<T> {
    const out = this.like(this.rows * r, this.cols * c);
    const R = this.rows, C = this.cols;
    for (let br = 0; br < r; br++) {
        for (let bc = 0; bc < c; bc++) {
            for (let i = 0; i < R; i++) {
                const srcOff = i * C;
                const dstOff = (br * R + i) * out.cols + bc * C;
                for (let j = 0; j < C; j++) {
                    out.data[dstOff + j] = this.data[srcOff + j];
                }
            }
        }
    }
    return out;
}

/** Slice: estrae una sotto-matrice [rowStart..rowEnd) × [colStart..colEnd). */
export function slice<T extends INumeric<T>>(
    this: Matrix<T>,
    rowStart: number,
    rowEnd: number,
    colStart: number,
    colEnd: number
): Matrix<T> {
    const nRows = rowEnd - rowStart;
    const nCols = colEnd - colStart;
    if (nRows <= 0 || nCols <= 0) throw new Error("slice: dimensioni non positive.");
    const out = this.like(nRows, nCols);
    let dst = 0;
    for (let i = rowStart; i < rowEnd; i++) {
        const rowOff = i * this.cols;
        for (let j = colStart; j < colEnd; j++) {
            out.data[dst++] = this.data[rowOff + j];
        }
    }
    return out;
}
