import { Matrix } from "..";

export function subtract(this: Matrix, B: Matrix | number): Matrix {
    if (typeof B === "number") return subScalar(this, B as number);
    
    if (this.rows === B.rows && this.cols === B.cols) return subMatrix(this, B);
    if (B.rows === 1 && B.cols === this.cols) return subRowVector(this, B);
    if (B.cols === 1 && B.rows === this.rows) return subColumnVector(this, B);

    throw new Error(`Incompatible dimensions for subtract: ${this.rows}x${this.cols} and ${B.rows}x${B.cols}`);
}

function subScalar(A: Matrix, scalar: number): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data, Cdat = out.data, len = Adat.length;
    let i = 0;
    for (; i <= len - 4; i += 4) {
        Cdat[i] = Adat[i] - scalar; Cdat[i+1] = Adat[i+1] - scalar;
        Cdat[i+2] = Adat[i+2] - scalar; Cdat[i+3] = Adat[i+3] - scalar;
    }
    for (; i < len; i++) Cdat[i] = Adat[i] - scalar;
    return out;
}

function subMatrix(A: Matrix, B: Matrix): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data, Bdat = B.data, Cdat = out.data, len = Adat.length;
    let i = 0;
    for (; i <= len - 4; i += 4) {
        Cdat[i] = Adat[i] - Bdat[i]; Cdat[i+1] = Adat[i+1] - Bdat[i+1];
        Cdat[i+2] = Adat[i+2] - Bdat[i+2]; Cdat[i+3] = Adat[i+3] - Bdat[i+3];
    }
    for (; i < len; i++) Cdat[i] = Adat[i] - Bdat[i];
    return out;
}

function subRowVector(A: Matrix, B: Matrix): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data, Bdat = B.data, Cdat = out.data;
    for (let i = 0; i < A.rows; i++) {
        const off = i * A.cols;
        let j = 0;
        for (; j <= A.cols - 4; j += 4) {
            Cdat[off+j] = Adat[off+j] - Bdat[j]; Cdat[off+j+1] = Adat[off+j+1] - Bdat[j+1];
            Cdat[off+j+2] = Adat[off+j+2] - Bdat[j+2]; Cdat[off+j+3] = Adat[off+j+3] - Bdat[j+3];
        }
        for (; j < A.cols; j++) Cdat[off+j] = Adat[off+j] - Bdat[j];
    }
    return out;
}

function subColumnVector(A: Matrix, B: Matrix): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data, Bdat = B.data, Cdat = out.data;
    for (let i = 0; i < A.rows; i++) {
        const off = i * A.cols;
        const bVal = Bdat[i];
        let j = 0;
        for (; j <= A.cols - 4; j += 4) {
            Cdat[off+j] = Adat[off+j] - bVal; Cdat[off+j+1] = Adat[off+j+1] - bVal;
            Cdat[off+j+2] = Adat[off+j+2] - bVal; Cdat[off+j+3] = Adat[off+j+3] - bVal;
        }
        for (; j < A.cols; j++) Cdat[off+j] = Adat[off+j] - bVal;
    }
    return out;
}