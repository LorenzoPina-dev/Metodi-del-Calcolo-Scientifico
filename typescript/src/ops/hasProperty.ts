// ops/hasProperty.ts
//
// Ottimizzazioni:
//  1. isUpperTriangular / isLowerTriangular: early-exit al primo elemento fuori posto.
//  2. isDiagonal: unica passata che controlla sia above che below.
//  3. isOrthogonal: early-exit appena un prodotto scalare diverge.
//  4. isZeroMatrix / hasFiniteValues: accesso diretto a data[].
//  5. Nessuna chiamata a get/set nel percorso caldo — usa data[] direttamente.
//
import { Matrix } from "..";
import { INumeric } from "../type";

const EPS = 1e-10;

export function isSquare<T extends INumeric<T>>(this: Matrix<T>): boolean {
    return this.rows === this.cols;
}

export function isSymmetric<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (this.rows !== this.cols) return false;
    const n = this.rows, d = this.data;
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            if (!d[i * n + j].subtract(d[j * n + i]).isNearZero(tol)) return false;
        }
    }
    return true;
}

export function isUpperTriangular<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const R = this.rows, C = this.cols, d = this.data;
    for (let i = 1; i < R; i++)
        for (let j = 0; j < Math.min(i, C); j++)
            if (!d[i * C + j].isNearZero(tol)) return false;
    return true;
}

export function isLowerTriangular<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const R = this.rows, C = this.cols, d = this.data;
    for (let i = 0; i < R; i++)
        for (let j = i + 1; j < C; j++)
            if (!d[i * C + j].isNearZero(tol)) return false;
    return true;
}

export function isDiagonal<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const R = this.rows, C = this.cols, d = this.data;
    for (let i = 0; i < R; i++)
        for (let j = 0; j < C; j++)
            if (i !== j && !d[i * C + j].isNearZero(tol)) return false;
    return true;
}

export function isIdentity<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (this.rows !== this.cols) return false;
    const n = this.rows, d = this.data;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const expected = i === j ? this.one : this.zero;
            if (!d[i * n + j].subtract(expected).isNearZero(tol)) return false;
        }
    }
    return true;
}

export function isOrthogonal<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (this.rows !== this.cols) return false;
    const n = this.rows, d = this.data;
    // Controlla che A^T * A = I colonna per colonna (evita la moltiplicazione completa)
    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            let dot = this.zero;
            for (let k = 0; k < n; k++) dot = dot.add(d[k * n + i].multiply(d[k * n + j]));
            const expected = i === j ? this.one : this.zero;
            if (!dot.subtract(expected).isNearZero(tol)) return false;
        }
    }
    return true;
}

export function isZeroMatrix<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const d = this.data, len = d.length;
    for (let i = 0; i < len; i++) if (!d[i].isNearZero(tol)) return false;
    return true;
}

export function isInvertible<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (this.rows !== this.cols) return false;
    try {
        const { U } = Matrix.decomp.lup(this);
        const n = U.rows;
        for (let i = 0; i < n; i++) if (U.data[i * n + i].isNearZero(tol)) return false;
        return true;
    } catch { return false; }
}

export function isSingular<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    return !this.isInvertible(tol);
}

export function isPositiveDefinite<T extends INumeric<T>>(this: Matrix<T>): boolean {
    if (!this.isSymmetric()) return false;
    try { Matrix.decomp.cholesky(this); return true; }
    catch { return false; }
}

export function isPositiveSemiDefinite<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (!this.isSymmetric()) return false;
    const n = this.rows, d = this.data, negTol = this.zero.fromNumber(-tol);
    for (let i = 0; i < n; i++)
        if (d[i * n + i].lessThan(negTol)) return false;
    return true;
}

export function isDiagonallyDominant<T extends INumeric<T>>(this: Matrix<T>): boolean {
    const R = this.rows, C = this.cols, d = this.data;
    for (let i = 0; i < R; i++) {
        const off = i * C;
        let offSum = this.zero;
        for (let j = 0; j < C; j++) if (j !== i) offSum = offSum.add(d[off + j].abs());
        if (d[off + i].abs().lessThan(offSum)) return false;
    }
    return true;
}

export function hasZeroTrace<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const n = Math.min(this.rows, this.cols), d = this.data, C = this.cols;
    let t = this.zero;
    for (let i = 0; i < n; i++) t = t.add(d[i * C + i]);
    return t.isNearZero(tol);
}

export function hasFiniteValues<T extends INumeric<T>>(this: Matrix<T>): boolean {
    const d = this.data, len = d.length;
    for (let i = 0; i < len; i++) if (!Number.isFinite(d[i].toNumber())) return false;
    return true;
}

export function isStochastic<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const R = this.rows, C = this.cols, d = this.data;
    const negTol = this.zero.fromNumber(-tol);
    for (let j = 0; j < C; j++) {
        let s = this.zero;
        for (let i = 0; i < R; i++) {
            const v = d[i * C + j];
            if (v.lessThan(negTol)) return false;
            s = s.add(v);
        }
        if (!s.subtract(this.one).isNearZero(tol)) return false;
    }
    return true;
}
