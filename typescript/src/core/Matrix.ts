import { lup } from "../decomposition";
import { identity, zeros } from "../init";
import { solve } from "../solver";

export class Matrix {
    data: Float64Array;
    rows: number;
    cols: number;

    constructor(rows: number, cols: number, data?: Float64Array) {
        this.rows = rows;
        this.cols = cols;
        this.data = data ?? new Float64Array(rows * cols);
    }

    // ---------------- INDEX ----------------
    private idx(i: number, j: number): number {
        return i * this.cols + j;
    }

    get(i: number, j: number): number {
        return this.data[this.idx(i, j)];
    }

    set(i: number, j: number, v: number): void {
        this.data[this.idx(i, j)] = v;
    }

    // ---------------- CONSTANTS ----------------
    static EPS = 1e-10;

    static isZero(x: number): boolean {
        return Math.abs(x) < Matrix.EPS;
    }

    // ---------------- COPY ----------------
    clone(): Matrix {
        return new Matrix(this.rows, this.cols, new Float64Array(this.data));
    }

    // ---------------- MATRICES ----------------
    triu(): Matrix {
        const result = zeros(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = i; j < this.cols; j++) result.set(i, j, this.get(i, j));
        }
        return result;
    }

    tril(): Matrix {
        const result = zeros(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j <= i; j++) result.set(i, j, this.get(i, j));
        }
        return result;
    }

    // ---------------- ALGEBRA LINEARE ----------------
    isSymmetric(): boolean {
        if (this.rows !== this.cols) return false;
        for (let i = 0; i < this.rows; i++) {
            for (let j = i + 1; j < this.cols; j++) {
                if (Math.abs(this.get(i, j) - this.get(j, i)) > Matrix.EPS) return false;
            }
        }
        return true;
    }

    add(B: Matrix): Matrix {
        const out = new Matrix(this.rows, this.cols);
        if (B.rows === 1 && B.cols === this.cols) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) + B.get(0, j));
        } else if (B.cols === 1 && B.rows === this.rows) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) + B.get(i, 0));
        } else if (B.rows === this.rows && B.cols === this.cols) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) + B.get(i, j));
        } else throw new Error("Incompatible shapes for add");
        return out;
    }

    subtract(B: Matrix): Matrix {
        const out = new Matrix(this.rows, this.cols);
        if (B.rows === 1 && B.cols === this.cols) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) - B.get(0, j));
        } else if (B.cols === 1 && B.rows === this.rows) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) - B.get(i, 0));
        } else if (B.rows === this.rows && B.cols === this.cols) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) - B.get(i, j));
        } else throw new Error("Incompatible shapes for subtract");
        return out;
    }

    multiply(B: Matrix): Matrix {
        if (this.cols !== B.rows) throw new Error("Dim mismatch");
        const out = new Matrix(this.rows, B.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let k = 0; k < this.cols; k++) {
                const aik = this.get(i, k);
                for (let j = 0; j < B.cols; j++) {
                    out.data[i * B.cols + j] += aik * B.get(k, j);
                }
            }
        }
        return out;
    }

    transpose(): Matrix {
        const out = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
                out.set(j, i, this.get(i, j));
        return out;
    }

    sum(axis: 0 | 1 = 0): Matrix {
        if (axis === 0) {
            const out = new Matrix(1, this.cols);
            for (let j = 0; j < this.cols; j++) {
                let s = 0;
                for (let i = 0; i < this.rows; i++) s += this.get(i, j);
                out.set(0, j, s);
            }
            return out;
        } else {
            const out = new Matrix(this.rows, 1);
            for (let i = 0; i < this.rows; i++) {
                let s = 0;
                for (let j = 0; j < this.cols; j++) s += this.get(i, j);
                out.set(i, 0, s);
            }
            return out;
        }
    }

    totalSum(): number {
        let s = 0;
        for (let i = 0; i < this.data.length; i++) s += this.data[i];
        return s;
    }

    max(): { value: number; row: number; col: number } {
        let maxVal = -Infinity, row = 0, col = 0;
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
                if (this.get(i, j) > maxVal) {
                    maxVal = this.get(i, j);
                    row = i; col = j;
                }
        return { value: maxVal, row, col };
    }

    slice(rowStart: number, rowEnd: number, colStart: number, colEnd: number): Matrix {
        const out = new Matrix(rowEnd - rowStart, colEnd - colStart);
        for (let i = rowStart; i < rowEnd; i++)
            for (let j = colStart; j < colEnd; j++)
                out.set(i - rowStart, j - colStart, this.get(i, j));
        return out;
    }

    // ---------------- DETERMINANTE E INVERSE ----------------
    det(): number {
        const { U, P } = lup(this);
        let det = 1, sign = 1;
        for (let i = 0; i < P.length; i++) if (P[i] !== i) sign *= -1;
        for (let i = 0; i < this.rows; i++) det *= U.get(i, i);
        return det * sign;
    }

    inverse(): Matrix {
        return solve(this, identity(this.rows));
    }

    // ---------------- STAMPA ----------------
    toString(): string {
        let s = "";
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) s += this.get(i, j).toFixed(3) + " ";
            s += "\n";
        }
        return s;
    }
}