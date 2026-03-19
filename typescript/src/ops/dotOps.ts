import { Matrix } from "..";

// --- DOT MULTIPLY (.*) ---
export function dotMultiply(this: Matrix, B: Matrix | number): Matrix {
    if (typeof B === "number") return applyScalar(this, B, (a, b) => a * b);
    return applyBroadcasting(this, B, (a, b) => a * b);
}

// --- DOT DIVIDE (./) ---
export function dotDivide(this: Matrix, B: Matrix | number): Matrix {
    if (typeof B === "number") {
        const inv = 1.0 / B;
        return applyScalar(this, inv, (a, b) => a * b);
    }
    return applyBroadcasting(this, B, (a, b) => a / b);
}

// --- DOT POWER (.^) ---
export function dotPow(this: Matrix, B: Matrix | number): Matrix {
    if (typeof B === "number") return applyScalar(this, B, Math.pow);
    return applyBroadcasting(this, B, Math.pow);
}

// --- HELPER GENERICI PER EVITARE DUPLICAZIONE DI CODICE ---

function applyScalar(A: Matrix, s: number, op: (a: number, b: number) => number): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data, Cdat = out.data;
    for (let i = 0; i < Adat.length; i++) Cdat[i] = op(Adat[i], s);
    return out;
}

function applyBroadcasting(A: Matrix, B: Matrix, op: (a: number, b: number) => number): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data, Bdat = B.data, Cdat = out.data;

    if (A.rows === B.rows && A.cols === B.cols) {
        for (let i = 0; i < Adat.length; i++) Cdat[i] = op(Adat[i], Bdat[i]);
    } else if (B.rows === 1 && B.cols === A.cols) { // Row Vector Broadcast
        for (let i = 0; i < A.rows; i++) {
            const off = i * A.cols;
            for (let j = 0; j < A.cols; j++) Cdat[off + j] = op(Adat[off + j], Bdat[j]);
        }
    } else if (B.cols === 1 && B.rows === A.rows) { // Col Vector Broadcast
        for (let i = 0; i < A.rows; i++) {
            const off = i * A.cols;
            for (let j = 0; j < A.cols; j++) Cdat[off + j] = op(Adat[off + j], Bdat[i]);
        }
    } else {
        throw new Error("Dimension mismatch for dot operation");
    }
    return out;
}