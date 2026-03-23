// ops/transform.ts — con fast-path WASM (trasposta cache-blocked)
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync } from "../wasm/wasm_bridge";

export function transpose<T extends INumeric<T>>(this: Matrix<T>): Matrix<T> {
    const R = this.rows, C = this.cols;
    if (this.isFloat64) {
        const w = getBridgeSync();
        if (w) {
            const aPtr = w.writeFloat64M(this.data as any);
            const cPtr = w.allocOutput(R * C);
            w.exports.transpose(aPtr, cPtr, R, C);
            const flat = w.readF64(cPtr, R * C);
            w.reset();
            const outData = new Array<T>(R * C);
            for (let i = 0; i < R * C; i++) outData[i] = this.zero.fromNumber(flat[i]);
            return this.likeWithData(C, R, outData);
        }
    }
    const ad = this.data;
    const outData = new Array<T>(R * C);
    for (let i = 0; i < R; i++) {
        const iOff = i * C;
        for (let j = 0; j < C; j++) outData[j * R + i] = ad[iOff + j];
    }
    return this.likeWithData(C, R, outData);
}

export function reshape<T extends INumeric<T>>(this: Matrix<T>, r: number, c: number): Matrix<T> {
    if (r * c !== this.rows * this.cols) throw new Error("reshape: numero elementi invariato.");
    return this.likeWithData(r, c, this.data.slice());
}

export function flip<T extends INumeric<T>>(this: Matrix<T>, dim: 1|2 = 1): Matrix<T> {
    const R = this.rows, C = this.cols;
    const ad = this.data;
    const outData = new Array<T>(R * C);
    if (dim === 1) {
        for (let i = 0; i < R; i++) {
            const src = i * C, dst = (R - 1 - i) * C;
            for (let j = 0; j < C; j++) outData[dst + j] = ad[src + j];
        }
    } else {
        for (let i = 0; i < R; i++) {
            const off = i * C;
            for (let j = 0; j < C; j++) outData[off + (C - 1 - j)] = ad[off + j];
        }
    }
    return this.likeWithData(R, C, outData);
}

export function rot90<T extends INumeric<T>>(this: Matrix<T>, k = 1): Matrix<T> {
    k = ((k % 4) + 4) % 4;
    if (k === 0) return this.clone();
    if (k === 2) return (flip<T>).call((flip<T>).call(this, 1), 2);
    const R = this.rows, C = this.cols;
    const ad = this.data;
    const outData = new Array<T>(R * C);
    if (k === 1) {
        for (let i = 0; i < R; i++)
            for (let j = 0; j < C; j++)
                outData[(C - 1 - j) * R + i] = ad[i * C + j];
    } else {
        for (let i = 0; i < R; i++)
            for (let j = 0; j < C; j++)
                outData[j * R + (R - 1 - i)] = ad[i * C + j];
    }
    return this.likeWithData(C, R, outData);
}

export function circshift<T extends INumeric<T>>(this: Matrix<T>, rShift: number, cShift: number): Matrix<T> {
    const R = this.rows, C = this.cols;
    const ad = this.data;
    rShift = ((rShift % R) + R) % R;
    cShift = ((cShift % C) + C) % C;
    const outData = new Array<T>(R * C);
    for (let i = 0; i < R; i++) {
        const ni = (i + rShift) % R;
        for (let j = 0; j < C; j++)
            outData[ni * C + ((j + cShift) % C)] = ad[i * C + j];
    }
    return this.likeWithData(R, C, outData);
}

export function repmat<T extends INumeric<T>>(this: Matrix<T>, r: number, c: number): Matrix<T> {
    const R = this.rows, C = this.cols;
    const OR = R * r, OC = C * c;
    const ad = this.data;
    const outData = new Array<T>(OR * OC);
    for (let br = 0; br < r; br++)
        for (let bc = 0; bc < c; bc++)
            for (let i = 0; i < R; i++) {
                const srcOff = i * C, dstOff = (br * R + i) * OC + bc * C;
                for (let j = 0; j < C; j++) outData[dstOff + j] = ad[srcOff + j];
            }
    return this.likeWithData(OR, OC, outData);
}

export function slice<T extends INumeric<T>>(
    this: Matrix<T>, rowStart: number, rowEnd: number, colStart: number, colEnd: number
): Matrix<T> {
    const nR = rowEnd - rowStart, nC = colEnd - colStart;
    if (nR <= 0 || nC <= 0) throw new Error("slice: dimensioni non positive.");
    const C = this.cols, ad = this.data;
    const outData = new Array<T>(nR * nC);
    let dst = 0;
    for (let i = rowStart; i < rowEnd; i++) {
        const rowOff = i * C;
        for (let j = colStart; j < colEnd; j++) outData[dst++] = ad[rowOff + j];
    }
    return this.likeWithData(nR, nC, outData);
}
