// ops/hasProperty.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

const EPS = 1e-10;

function _wasmBool<T extends INumeric<T>>(
    A: Matrix<T>, minElems: number,
    fn: (w: NonNullable<ReturnType<typeof getBridgeSync>>, aPtr: number) => number
): boolean | null {
    if (!A.isFloat64 || A.data.length < minElems) return null;
    const w = getBridgeSync();
    if (!w) return null;
    const aPtr = w.writeFloat64M(A.data as any);
    const res = fn(w, aPtr);
    w.reset();
    return res !== 0;
}

export function isSquare<T extends INumeric<T>>(this: Matrix<T>): boolean { return this.rows === this.cols; }

export function isSymmetric<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (this.rows !== this.cols) return false;
    const r = _wasmBool(this, WASM_THRESHOLD.PROPERTY, (w,p) => w.exports.isSymmetricF64(p, this.rows, tol));
    if (r !== null) return r;
    const n = this.rows, d = this.data;
    for (let i=0;i<n;i++) for (let j=i+1;j<n;j++) if (!d[i*n+j].subtract(d[j*n+i]).isNearZero(tol)) return false;
    return true;
}

export function isUpperTriangular<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const r = _wasmBool(this, WASM_THRESHOLD.PROPERTY, (w,p) => w.exports.isUpperTriF64(p, this.rows, this.cols, tol));
    if (r !== null) return r;
    const R=this.rows, C=this.cols, d=this.data;
    for (let i=1;i<R;i++) for (let j=0;j<Math.min(i,C);j++) if (!d[i*C+j].isNearZero(tol)) return false;
    return true;
}

export function isLowerTriangular<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const r = _wasmBool(this, WASM_THRESHOLD.PROPERTY, (w,p) => w.exports.isLowerTriF64(p, this.rows, this.cols, tol));
    if (r !== null) return r;
    const R=this.rows, C=this.cols, d=this.data;
    for (let i=0;i<R;i++) for (let j=i+1;j<C;j++) if (!d[i*C+j].isNearZero(tol)) return false;
    return true;
}

export function isDiagonal<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const r = _wasmBool(this, WASM_THRESHOLD.PROPERTY, (w,p) => w.exports.isDiagonalF64(p, this.rows, this.cols, tol));
    if (r !== null) return r;
    const R=this.rows, C=this.cols, d=this.data;
    for (let i=0;i<R;i++) for (let j=0;j<C;j++) if (i!==j && !d[i*C+j].isNearZero(tol)) return false;
    return true;
}

export function isIdentity<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (this.rows !== this.cols) return false;
    const n=this.rows, d=this.data;
    for (let i=0;i<n;i++) for (let j=0;j<n;j++) {
        const expected = i===j ? this.one : this.zero;
        if (!d[i*n+j].subtract(expected).isNearZero(tol)) return false;
    }
    return true;
}

export function isOrthogonal<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (this.rows !== this.cols) return false;
    const n=this.rows, d=this.data;
    for (let i=0;i<n;i++) for (let j=i;j<n;j++) {
        let dot = this.zero;
        for (let k=0;k<n;k++) dot = dot.add(d[k*n+i].multiply(d[k*n+j]));
        if (!dot.subtract(i===j?this.one:this.zero).isNearZero(tol)) return false;
    }
    return true;
}

export function isZeroMatrix<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const r = _wasmBool(this, WASM_THRESHOLD.PROPERTY, (w,p) => w.exports.isZeroF64(p, this.data.length, tol));
    if (r !== null) return r;
    const d=this.data, len=d.length;
    for (let i=0;i<len;i++) if (!d[i].isNearZero(tol)) return false;
    return true;
}

export function isInvertible<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (this.rows !== this.cols) return false;
    try {
        const { U } = Matrix.decomp.lup(this);
        const n=U.rows;
        for (let i=0;i<n;i++) if (U.data[i*n+i].isNearZero(tol)) return false;
        return true;
    } catch { return false; }
}

export function isSingular<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean { return !this.isInvertible(tol); }

export function isPositiveDefinite<T extends INumeric<T>>(this: Matrix<T>): boolean {
    if (!this.isSymmetric()) return false;
    try { Matrix.decomp.cholesky(this); return true; } catch { return false; }
}

export function isPositiveSemiDefinite<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    if (!this.isSymmetric()) return false;
    const n=this.rows, d=this.data, negTol=this.zero.fromNumber(-tol);
    for (let i=0;i<n;i++) if (d[i*n+i].lessThan(negTol)) return false;
    return true;
}

export function isDiagonallyDominant<T extends INumeric<T>>(this: Matrix<T>): boolean {
    const r = _wasmBool(this, WASM_THRESHOLD.PROPERTY, (w,p) => w.exports.isDiagDomF64(p, this.rows, this.cols));
    if (r !== null) return r;
    const R=this.rows, C=this.cols, d=this.data;
    for (let i=0;i<R;i++) {
        const off=i*C; let s=this.zero;
        for (let j=0;j<C;j++) if (j!==i) s=s.add(d[off+j].abs());
        if (d[off+i].abs().lessThan(s)) return false;
    }
    return true;
}

export function hasZeroTrace<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const n=Math.min(this.rows,this.cols), d=this.data, C=this.cols;
    let t=this.zero;
    for (let i=0;i<n;i++) t=t.add(d[i*C+i]);
    return t.isNearZero(tol);
}

export function hasFiniteValues<T extends INumeric<T>>(this: Matrix<T>): boolean {
    const r = _wasmBool(this, WASM_THRESHOLD.PROPERTY, (w,p) => w.exports.hasFiniteF64(p, this.data.length));
    if (r !== null) return r;
    const d=this.data, len=d.length;
    for (let i=0;i<len;i++) if (!Number.isFinite(d[i].toNumber())) return false;
    return true;
}

export function isStochastic<T extends INumeric<T>>(this: Matrix<T>, tol = EPS): boolean {
    const R=this.rows, C=this.cols, d=this.data, negTol=this.zero.fromNumber(-tol);
    for (let j=0;j<C;j++) {
        let s=this.zero;
        for (let i=0;i<R;i++) { const v=d[i*C+j]; if (v.lessThan(negTol)) return false; s=s.add(v); }
        if (!s.subtract(this.one).isNearZero(tol)) return false;
    }
    return true;
}
