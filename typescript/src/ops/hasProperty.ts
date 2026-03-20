// ops/hasProperty.ts

import { Matrix } from "..";

// ---------------- QUADRATA ----------------
export function isSquare(this: Matrix): boolean {
    return this.rows === this.cols;
}

// ---------------- SIMMETRICA ----------------
export function isSymmetric(this: Matrix, tol: number = 1e-10): boolean {
    if (!this.isSquare()) return false;

    for (let i = 0; i < this.rows; i++) {
        for (let j = i + 1; j < this.cols; j++) {
            if (Math.abs(this.get(i, j) - this.get(j, i)) > tol) return false;
        }
    }
    return true;
}

// ---------------- TRIANGOLARI ----------------
export function isUpperTriangular(this: Matrix, tol = 1e-10): boolean {
    for (let i = 1; i < this.rows; i++) {
        for (let j = 0; j < i; j++) {
            if (Math.abs(this.get(i, j)) > tol) return false;
        }
    }
    return true;
}

export function isLowerTriangular(this: Matrix, tol = 1e-10): boolean {
    for (let i = 0; i < this.rows; i++) {
        for (let j = i + 1; j < this.cols; j++) {
            if (Math.abs(this.get(i, j)) > tol) return false;
        }
    }
    return true;
}

// ---------------- DIAGONALE ----------------
export function isDiagonal(this: Matrix, tol = 1e-10): boolean {
    for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
            if (i !== j && Math.abs(this.get(i, j)) > tol) return false;
        }
    }
    return true;
}

// ---------------- IDENTITÀ ----------------
export function isIdentity(this: Matrix, tol = 1e-10): boolean {
    if (!this.isSquare()) return false;

    for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
            const expected = i === j ? 1 : 0;
            if (Math.abs(this.get(i, j) - expected) > tol) return false;
        }
    }
    return true;
}

// ---------------- ORTOGONALE ----------------
export function isOrthogonal(this: Matrix, tol = 1e-10): boolean {
    if (!this.isSquare()) return false;

    const n = this.rows;

    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {

            let dot = 0;
            for (let k = 0; k < n; k++) {
                dot += this.get(k, i) * this.get(k, j);
            }

            const expected = i === j ? 1 : 0;
            if (Math.abs(dot - expected) > tol) return false;
        }
    }

    return true;
}

// ---------------- ZERO ----------------
export function isZeroMatrix(this: Matrix, tol = 1e-10): boolean {
    for (let i = 0; i < this.data.length; i++) {
        if (Math.abs(this.data[i]) > tol) return false;
    }
    return true;
}

// ================= NUOVI METODI =================

// ---------------- INVERTIBILE ----------------
export function isInvertible(this: Matrix, tol = 1e-12): boolean {
    if (!this.isSquare()) return false;

    try {
        const { U } = Matrix.decomp.lup(this);
        for (let i = 0; i < U.rows; i++) {
            if (Math.abs(U.get(i, i)) < tol) return false;
        }
        return true;
    } catch {
        return false;
    }
}

// ---------------- SINGOLARE ----------------
export function isSingular(this: Matrix, tol = 1e-12): boolean {
    return !this.isInvertible(tol);
}

// ---------------- POSITIVA DEFINITA ----------------
export function isPositiveDefinite(this: Matrix): boolean {
    if (!this.isSymmetric()) return false;

    try {
        Matrix.decomp.cholesky(this);
        return true;
    } catch {
        return false;
    }
}

// ---------------- SEMI-DEFINITA ----------------
export function isPositiveSemiDefinite(this: Matrix, tol = 1e-12): boolean {
    if (!this.isSymmetric()) return false;

    const n = this.rows;

    for (let i = 0; i < n; i++) {
        if (this.get(i, i) < -tol) return false;
    }

    return true;
}

// ---------------- DOMINANTE DIAGONALE ----------------
export function isDiagonallyDominant(this: Matrix): boolean {
    for (let i = 0; i < this.rows; i++) {
        let sum = 0;
        for (let j = 0; j < this.cols; j++) {
            if (i !== j) sum += Math.abs(this.get(i, j));
        }

        if (Math.abs(this.get(i, i)) < sum) return false;
    }
    return true;
}

// ---------------- TRACCIA ZERO ----------------
export function hasZeroTrace(this: Matrix, tol = 1e-10): boolean {
    let t = 0;
    const n = Math.min(this.rows, this.cols);

    for (let i = 0; i < n; i++) {
        t += this.get(i, i);
    }

    return Math.abs(t) < tol;
}

// ---------------- NORMA FINITA ----------------
export function hasFiniteValues(this: Matrix): boolean {
    for (let i = 0; i < this.data.length; i++) {
        if (!Number.isFinite(this.data[i])) return false;
    }
    return true;
}

// ---------------- MATRICE STOCASTICA ----------------
export function isStochastic(this: Matrix, tol = 1e-10): boolean {
    for (let j = 0; j < this.cols; j++) {
        let sum = 0;
        for (let i = 0; i < this.rows; i++) {
            const v = this.get(i, j);
            if (v < -tol) return false;
            sum += v;
        }
        if (Math.abs(sum - 1) > tol) return false;
    }
    return true;
}