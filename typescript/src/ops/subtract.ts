// ops/subtract.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function subtract<T extends INumeric<T>>(this: Matrix<T>, B: Matrix<T> | number): Matrix<T> {
    if (typeof B === "number") return _subScalar(this, B);
    if (this.rows === B.rows && this.cols === B.cols) return _subMatrix(this, B);
    if (B.rows === 1 && B.cols === this.cols) return _subRowVec(this, B);
    if (B.cols === 1 && B.rows === this.rows) return _subColVec(this, B);
    throw new Error(`subtract: dimensioni incompatibili ${this.rows}×${this.cols} e ${B.rows}×${B.cols}`);
}

function _subScalar<T extends INumeric<T>>(A: Matrix<T>, scalar: number): Matrix<T> {
    const d = A.data, len = d.length;
    const outData = new Array<T>(len);
    if (A.isFloat64) {
        for (let i = 0; i < len; i++) outData[i] = A.zero.fromNumber((d[i] as any).value - scalar);
    } else {
        const s = A.zero.fromNumber(scalar);
        for (let i = 0; i < len; i++) outData[i] = d[i].subtract(s);
    }
    return A.likeWithData(A.rows, A.cols, outData);
}

function _subMatrix<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const ad = A.data, bd = B.data, len = ad.length;
    const outData = new Array<T>(len);
    if (A.isFloat64 && B.isFloat64) {
        for (let i = 0; i < len; i++)
            outData[i] = A.zero.fromNumber((ad[i] as any).value - (bd[i] as any).value);
    } else {
        for (let i = 0; i < len; i++) outData[i] = ad[i].subtract(bd[i]);
    }
    return A.likeWithData(A.rows, A.cols, outData);
}

function _subRowVec<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const R = A.rows, C = A.cols;
    const ad = A.data, bd = B.data;
    const outData = new Array<T>(R * C);
    if (A.isFloat64 && B.isFloat64) {
        for (let i = 0; i < R; i++) {
            const off = i * C;
            for (let j = 0; j < C; j++)
                outData[off + j] = A.zero.fromNumber((ad[off + j] as any).value - (bd[j] as any).value);
        }
    } else {
        for (let i = 0; i < R; i++) {
            const off = i * C;
            for (let j = 0; j < C; j++) outData[off + j] = ad[off + j].subtract(bd[j]);
        }
    }
    return A.likeWithData(R, C, outData);
}

function _subColVec<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>): Matrix<T> {
    const R = A.rows, C = A.cols;
    const ad = A.data, bd = B.data;
    const outData = new Array<T>(R * C);
    if (A.isFloat64 && B.isFloat64) {
        for (let i = 0; i < R; i++) {
            const off = i * C, bv = (bd[i] as any).value as number;
            for (let j = 0; j < C; j++)
                outData[off + j] = A.zero.fromNumber((ad[off + j] as any).value - bv);
        }
    } else {
        for (let i = 0; i < R; i++) {
            const off = i * C, bv = bd[i];
            for (let j = 0; j < C; j++) outData[off + j] = ad[off + j].subtract(bv);
        }
    }
    return A.likeWithData(R, C, outData);
}
