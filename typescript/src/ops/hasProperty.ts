// ops/hasProperty.ts
import { Matrix } from "..";
import { INumeric } from "../type";

const EPS = 1e-10;

// ---- QUADRATA ----
export function isSquare<T extends INumeric<T>>(this: Matrix<T>): boolean {
    return this.rows === this.cols;
}

// ---- SIMMETRICA ----
export function isSymmetric<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (!this.isSquare()) return false;
    for (let i = 0; i < this.rows; i++) {
        for (let j = i + 1; j < this.cols; j++) {
            if (!this.get(i, j).subtract(this.get(j, i)).isNearZero(tol)) return false;
        }
    }
    return true;
}

// ---- TRIANGOLARI ----
export function isUpperTriangular<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    for (let i = 1; i < this.rows; i++) {
        for (let j = 0; j < i; j++) {
            if (!this.get(i, j).isNearZero(tol)) return false;
        }
    }
    return true;
}

export function isLowerTriangular<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    for (let i = 0; i < this.rows; i++) {
        for (let j = i + 1; j < this.cols; j++) {
            if (!this.get(i, j).isNearZero(tol)) return false;
        }
    }
    return true;
}

// ---- DIAGONALE ----
export function isDiagonal<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
            if (i !== j && !this.get(i, j).isNearZero(tol)) return false;
        }
    }
    return true;
}

// ---- IDENTITÀ ----
export function isIdentity<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (!this.isSquare()) return false;
    for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
            const expected = i === j ? this.one : this.zero;
            if (!this.get(i, j).subtract(expected).isNearZero(tol)) return false;
        }
    }
    return true;
}

// ---- ORTOGONALE ----
export function isOrthogonal<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (!this.isSquare()) return false;
    const n = this.rows;
    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            let dot = this.zero;
            for (let k = 0; k < n; k++) {
                dot = dot.add(this.get(k, i).multiply(this.get(k, j)));
            }
            const expected = i === j ? this.one : this.zero;
            if (!dot.subtract(expected).isNearZero(tol)) return false;
        }
    }
    return true;
}

// ---- ZERO ----
export function isZeroMatrix<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    for (let i = 0; i < this.data.length; i++) {
        if (!this.data[i].isNearZero(tol)) return false;
    }
    return true;
}

// ---- INVERTIBILE ----
export function isInvertible<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (!this.isSquare()) return false;
    try {
        const { U } = Matrix.decomp.lup(this);
        for (let i = 0; i < U.rows; i++) {
            if (U.get(i, i).isNearZero(tol)) return false;
        }
        return true;
    } catch { return false; }
}

// ---- SINGOLARE ----
export function isSingular<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    return !this.isInvertible(tol);
}

// ---- POSITIVA DEFINITA ----
export function isPositiveDefinite<T extends INumeric<T>>(this: Matrix<T>): boolean {
    if (!this.isSymmetric()) return false;
    try { Matrix.decomp.cholesky(this); return true; }
    catch { return false; }
}

// ---- SEMI-DEFINITA ----
export function isPositiveSemiDefinite<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (!this.isSymmetric()) return false;
    for (let i = 0; i < this.rows; i++) {
        // diag element deve essere ≥ 0 (approssimazione)
        if (this.get(i, i).negate().greaterThan(this.zero.fromNumber(tol))) return false;
    }
    return true;
}

// ---- DOMINANTE DIAGONALE ----
export function isDiagonallyDominant<T extends INumeric<T>>(this: Matrix<T>): boolean {
    for (let i = 0; i < this.rows; i++) {
        let offSum = this.zero;
        for (let j = 0; j < this.cols; j++) {
            if (i !== j) offSum = offSum.add(this.get(i, j).abs());
        }
        if (this.get(i, i).abs().lessThan(offSum)) return false;
    }
    return true;
}

// ---- TRACCIA ZERO ----
export function hasZeroTrace<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    let t = this.zero;
    for (let i = 0; i < Math.min(this.rows, this.cols); i++) {
        t = t.add(this.get(i, i));
    }
    return t.isNearZero(tol);
}

// ---- VALORI FINITI ----
export function hasFiniteValues<T extends INumeric<T>>(this: Matrix<T>): boolean {
    for (let i = 0; i < this.data.length; i++) {
        if (!Number.isFinite(this.data[i].toNumber())) return false;
    }
    return true;
}

// ---- STOCASTICA ----
export function isStochastic<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    for (let j = 0; j < this.cols; j++) {
        let s = this.zero;
        for (let i = 0; i < this.rows; i++) {
            const v = this.get(i, j);
            if (v.negate().greaterThan(this.zero.fromNumber(tol))) return false;
            s = s.add(v);
        }
        if (!s.subtract(this.one).isNearZero(tol)) return false;
    }
    return true;
}
