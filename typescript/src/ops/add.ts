// ops/add.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function add<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") return _addScalar(this, B);
    if (this.rows === B.rows && this.cols === B.cols) return _addMatrix(this, B);
    if (B.rows === 1 && B.cols === this.cols) return _addRowVec(this, B);
    if (B.cols === 1 && B.rows === this.rows) return _addColVec(this, B);
    throw new Error(`add: dimensioni incompatibili ${this.rows}×${this.cols} e ${B.rows}×${B.cols}`);
}

export function totalSum<T extends INumeric<T>>(this: Matrix<T>): T {
    const d = this.data, len = d.length;
    let s = this.zero;
    for (let i = 0; i < len; i++) s = s.add(d[i]);
    return s;
}

// ---- helper ----

function _addScalar<T extends INumeric<T>>(A: Matrix<T>, scalar: number): Matrix<T> {
    const d = A.data, len = d.length;
    const outData = new Array<T>(len);
    if (A.isFloat64) {
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
    if (A.isFloat64 && B.isFloat64) {
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
    if (A.isFloat64 && B.isFloat64) {
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
    if (A.isFloat64 && B.isFloat64) {
        for (let i = 0; i < R; i++) {
            const off = i * C, bv = (bd[i] as any).value as number;
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
