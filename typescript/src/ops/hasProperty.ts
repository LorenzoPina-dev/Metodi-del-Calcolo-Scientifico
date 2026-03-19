// ops/hasProperty.ts

import { Matrix } from "..";

// ---------------- QUADRATA ----------------
export function isSquare(this: Matrix): boolean {
    return this.rows === this.cols;
}

// ---------------- SIMMETRICA ----------------
export function isSymmetric(this: Matrix, tol: number = 1e-10): boolean {
    if (this.rows !== this.cols) return false;

    for (let i = 0; i < this.rows; i++) {
        for (let j = i + 1; j < this.cols; j++) {
            if (Math.abs(this.get(i, j) - this.get(j, i)) > tol) {
                return false;
            }
        }
    }

    return true;
}

// ---------------- TRIANGOLARE SUPERIORE ----------------
export function isUpperTriangular(this: Matrix, tol: number = 1e-10): boolean {
    for (let i = 1; i < this.rows; i++) {
        for (let j = 0; j < i; j++) {
            if (Math.abs(this.get(i, j)) > tol) return false;
        }
    }
    return true;
}

// ---------------- TRIANGOLARE INFERIORE ----------------
export function isLowerTriangular(this: Matrix, tol: number = 1e-10): boolean {
    for (let i = 0; i < this.rows; i++) {
        for (let j = i + 1; j < this.cols; j++) {
            if (Math.abs(this.get(i, j)) > tol) return false;
        }
    }
    return true;
}

// ---------------- DIAGONALE ----------------
export function isDiagonal(this: Matrix, tol: number = 1e-10): boolean {
    for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
            if (i !== j && Math.abs(this.get(i, j)) > tol) {
                return false;
            }
        }
    }
    return true;
}

// ---------------- IDENTITÀ ----------------
export function isIdentity(this: Matrix, tol: number = 1e-10): boolean {
    if (this.rows !== this.cols) return false;

    for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
            const expected = (i === j) ? 1 : 0;
            if (Math.abs(this.get(i, j) - expected) > tol) {
                return false;
            }
        }
    }

    return true;
}

// ---------------- ORTOGONALE ----------------
export function isOrthogonal(this: Matrix, tol: number = 1e-10): boolean {
    if (this.rows !== this.cols) return false;

    const n = this.rows;

    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {

            let dot = 0;

            for (let k = 0; k < n; k++) {
                dot += this.get(k, i) * this.get(k, j);
            }

            const expected = (i === j) ? 1 : 0;

            if (Math.abs(dot - expected) > tol) {
                return false;
            }
        }
    }

    return true;
}

// ---------------- ZERO MATRIX ----------------
export function isZeroMatrix(this: Matrix, tol: number = 1e-10): boolean {
    for (let i = 0; i < this.data.length; i++) {
        if (Math.abs(this.data[i]) > tol) return false;
    }
    return true;
}