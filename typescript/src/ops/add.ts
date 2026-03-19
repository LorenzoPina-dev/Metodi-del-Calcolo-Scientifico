import { Matrix } from "..";


export function totalSum(this: Matrix): number {
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += this.data[i];
    return s;
}

/**
 * Somma universale (A + B). 
 * Gestisce: Scalari, Matrici identiche, Vettori riga (1xN), Vettori colonna (Mx1).
 */
export function add(this: Matrix, B: Matrix | number): Matrix {
    const out = new Matrix(this.rows, this.cols);

    if (typeof B === "number") return addScalar(this, B as number);
    if (this.rows === B.rows && this.cols === B.cols) return addMatrix(this, B);
    if (B.rows === 1 && B.cols === this.cols) return addRowVector(this, B);
    if (B.cols === 1 && B.rows === this.rows) return addColumnVector(this, B);

    throw new Error(`Incompatible dimensions for add: ${this.rows}x${this.cols} and ${B.rows}x${B.cols}`);
}

function addScalar(A: Matrix, scalar: number): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data,Cdat = out.data,len = Adat.length;
    let i = 0;
    for (; i <= len - 4; i += 4) {
        Cdat[i] = Adat[i] + scalar;
        Cdat[i + 1] = Adat[i + 1] + scalar;
        Cdat[i + 2] = Adat[i + 2] + scalar;
        Cdat[i + 3] = Adat[i + 3] + scalar;
    }
    for (; i < len; i++) Cdat[i] = Adat[i] + scalar;
    return out;
}

function addMatrix(A: Matrix, B: Matrix): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data,Cdat = out.data, Bdat = B.data, len = Adat.length;
    let i = 0;
        for (; i <= len - 4; i += 4) {
            Cdat[i] = Adat[i] + Bdat[i];
            Cdat[i + 1] = Adat[i + 1] + Bdat[i + 1];
            Cdat[i + 2] = Adat[i + 2] + Bdat[i + 2];
            Cdat[i + 3] = Adat[i + 3] + Bdat[i + 3];
        }
        for (; i < len; i++) Cdat[i] = Adat[i] + Bdat[i];
        return out;
}

function addRowVector(A: Matrix, B: Matrix): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data, Cdat = out.data, Bdat = B.data;
    for (let i = 0; i < A.rows; i++) {
        const offset = i * A.cols;
        let j = 0;
        // Unrolling sulle colonne
        for (; j <= A.cols - 4; j += 4) {
            Cdat[offset + j] = Adat[offset + j] + Bdat[j];
            Cdat[offset + j + 1] = Adat[offset + j + 1] + Bdat[j + 1];
            Cdat[offset + j + 2] = Adat[offset + j + 2] + Bdat[j + 2];
            Cdat[offset + j + 3] = Adat[offset + j + 3] + Bdat[j + 3];
        }
        for (; j < A.cols; j++) Cdat[offset + j] = Adat[offset + j] + Bdat[j];
    }
    return out;
}
function addColumnVector(A: Matrix, B: Matrix): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data, Cdat = out.data, Bdat = B.data;
    for (let i = 0; i < A.rows; i++) {
            const offset = i * A.cols;
            const bVal = Bdat[i]; // Carichiamo il valore della colonna nel registro una volta sola
            let j = 0;
            for (; j <= A.cols - 4; j += 4) {
                Cdat[offset + j] = Adat[offset + j] + bVal;
                Cdat[offset + j + 1] = Adat[offset + j + 1] + bVal;
                Cdat[offset + j + 2] = Adat[offset + j + 2] + bVal;
                Cdat[offset + j + 3] = Adat[offset + j + 3] + bVal;
            }
            for (; j < A.cols; j++) Cdat[offset + j] = Adat[offset + j] + bVal;
        }
        return out;
}

