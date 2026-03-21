// Matrix.ts
import { MatrixBase }   from "./core/MatrixBase";
import { INumeric, Float64M } from "./type";

import * as addOps   from "./ops/add";
import * as subOps   from "./ops/subtract";
import * as mulOps   from "./ops/multiply";
import * as dotOps   from "./ops/dotOps";
import * as statOps  from "./ops/statistics";
import * as unaryOps from "./ops/unary";
import * as transOps from "./ops/transform";
import * as propOps  from "./ops/hasProperty";
import { equal }     from "./ops/equal";
import { det }       from "./ops/det";
import { norm }      from "./ops/norm";
import { pow }       from "./ops/pow";

import * as gallery     from "./init/known";
import * as solver      from "./solver";
import * as decomp      from "./decomposition";
import { smartInverse } from "./algoritm/inverse";

import { zeros, ones, identity, diag, diagFromArray, zerosLike, identityLike } from "./init/init";
import { hankel, random, sparse, toeplitz, vander } from "./init";
import { tril, triu } from "./decomposition";

export class Matrix<T extends INumeric<T> = Float64M> extends MatrixBase<T> {

    constructor(
        rows: number,
        cols: number,
        zeroOrData?: T | Float64Array | ArrayLike<number> | Array<number>,
        one?: T,
        data?: Array<T>
    ) {
        if (zeroOrData === undefined || zeroOrData === null) {
            const f0 = Float64M.zero as unknown as T;
            super(rows, cols, f0, Float64M.one as unknown as T);
        } else if (zeroOrData instanceof Float64Array || Array.isArray(zeroOrData)) {
            const f0 = Float64M.zero as unknown as T;
            const wrapped = Array.from(zeroOrData as ArrayLike<number>)
                .map(v => new Float64M(v) as unknown as T);
            super(rows, cols, f0, Float64M.one as unknown as T, wrapped);
        } else {
            super(rows, cols, zeroOrData as T, one!, data);
        }
    }

    // ---- Clone / factory ottimizzati ----
    override clone(): this {
        return new Matrix<T>(this.rows, this.cols, this.zero, this.one, this.data.slice()) as this;
    }
    override like(rows: number, cols: number): this {
        return new Matrix<T>(rows, cols, this.zero, this.one) as this;
    }
    override likeWithData(rows: number, cols: number, data: Array<T>): this {
        return new Matrix<T>(rows, cols, this.zero, this.one, data) as this;
    }

    // ---- get / set backward-compat ----
    override get(i: number, j: number): T { return this.data[i * this.cols + j]; }
    getNum(i: number, j: number): number  { return this.data[i * this.cols + j].toNumber(); }
    override set(i: number, j: number, v: T | number): void {
        this.data[i * this.cols + j] = typeof v === "number"
            ? this.zero.fromNumber(v) : (v as T);
    }

    // -------- UGUAGLIANZA --------
    equals(B: Matrix<T>, tol = MatrixBase.EPS): boolean { return equal(this, B, tol); }

    // -------- ARITMETICA --------
    add(B: Matrix<T> | number): Matrix<T>  { return addOps.add.call(this, B); }
    sub(B: Matrix<T> | number): Matrix<T>  { return subOps.subtract.call(this, B); }
    mul(B: Matrix<T> | number): Matrix<T>  { return mulOps.multiply.call(this, B); }
    pow(exp: number): Matrix<T>            { return pow.call(this, exp); }

    // -------- DOT OPS --------
    dotMul(B: Matrix<T> | number): Matrix<T>   { return dotOps.dotMultiply.call(this, B); }
    dotDiv(B: Matrix<T> | number): Matrix<T>   { return dotOps.dotDivide.call(this, B); }
    dotPow(e: number | Matrix<T>): Matrix<T>   { return dotOps.dotPow.call(this, e); }

    // -------- ALGEBRA --------
    det(): T                   { return det(this); }
    norm(type?: any): number   { return norm.call(this, type); }
    inv(): Matrix<T>           { return smartInverse(this); }
    inverse(): Matrix<T>       { return this.inv(); }
    trace(): T                 { return unaryOps.trace.call(this); }
    totalSum(): T              { return addOps.totalSum.call(this); }

    // -------- STATISTICHE --------
    sum(dim: 1 | 2 = 1): Matrix<T>  { return statOps.sum.call(this, dim); }
    max(dim: 1 | 2 = 1)             { return statOps.max.call(this, dim); }
    min(dim: 1 | 2 = 1)             { return statOps.min.call(this, dim); }
    mean(dim: 1 | 2 = 1): Matrix<T> { return statOps.mean.call(this, dim); }

    // -------- TRASFORMAZIONI --------
    t(): Matrix<T>                                                    { return transOps.transpose.call(this); }
    /**
     * Trasposta coniugata / Hermitian adjoint: A^H.
     *   - Per Float64M e Rational: uguale a t() (i reali sono auto-coniugati).
     *   - Per Complex: (A^H)_{ij} = conj(A_{ji}).
     * Usare sempre ct() nel solver QR e in Cholesky per garantire correttezza
     * anche su matrici complesse.
     */
    override ct(): this                                               { return super.ct(); }
    reshape(r: number, c: number): Matrix<T>                          { return transOps.reshape.call(this, r, c); }
    repmat(r: number, c: number): Matrix<T>                           { return transOps.repmat.call(this, r, c); }
    flip(dim: 1 | 2 = 1): Matrix<T>                                   { return transOps.flip.call(this, dim); }
    rot90(k = 1): Matrix<T>                                           { return transOps.rot90.call(this, k); }
    slice(rs: number, re: number, cs: number, ce: number): Matrix<T>  { return transOps.slice.call(this, rs, re, cs, ce); }

    // -------- UNARIE --------
    abs(): Matrix<T>    { return unaryOps.abs.call(this); }
    sqrt(): Matrix<T>   { return unaryOps.sqrt.call(this); }
    round(): Matrix<T>  { return unaryOps.round.call(this); }
    negate(): Matrix<T> { return unaryOps.negate.call(this); }
    exp(): Matrix<T>    { return unaryOps.exp.call(this); }
    floor(): Matrix<T>  { return unaryOps.floor.call(this); }
    ceil(): Matrix<T>   { return unaryOps.ceil.call(this); }
    sin(): Matrix<T>    { return unaryOps.sin.call(this); }
    cos(): Matrix<T>    { return unaryOps.cos.call(this); }
    tan(): Matrix<T>    { return unaryOps.tan.call(this); }

    // -------- PROPRIETÀ --------
    isSquare(): boolean                           { return propOps.isSquare.call(this); }
    isSymmetric(tol?: number): boolean            { return propOps.isSymmetric.call(this, tol); }
    isUpperTriangular(tol?: number): boolean      { return propOps.isUpperTriangular.call(this, tol); }
    isLowerTriangular(tol?: number): boolean      { return propOps.isLowerTriangular.call(this, tol); }
    isDiagonal(tol?: number): boolean             { return propOps.isDiagonal.call(this, tol); }
    isIdentity(tol?: number): boolean             { return propOps.isIdentity.call(this, tol); }
    isOrthogonal(tol?: number): boolean           { return propOps.isOrthogonal.call(this, tol); }
    isZeroMatrix(tol?: number): boolean           { return propOps.isZeroMatrix.call(this, tol); }
    isInvertible(tol?: number): boolean           { return propOps.isInvertible.call(this, tol); }
    isSingular(tol?: number): boolean             { return propOps.isSingular.call(this, tol); }
    isPositiveDefinite(): boolean                 { return propOps.isPositiveDefinite.call(this); }
    isPositiveSemiDefinite(tol?: number): boolean { return propOps.isPositiveSemiDefinite.call(this, tol); }
    isDiagonallyDominant(): boolean               { return propOps.isDiagonallyDominant.call(this); }
    hasZeroTrace(tol?: number): boolean           { return propOps.hasZeroTrace.call(this, tol); }
    hasFiniteValues(): boolean                    { return propOps.hasFiniteValues.call(this); }
    isStochastic(tol?: number): boolean           { return propOps.isStochastic.call(this, tol); }

    // -------- SOLVER --------
    solve(b: Matrix<T>, method = "LUP"): Matrix<T> { return solver.solve.call(this, b, method); }

    // -------- STATIC NAMESPACE --------
    static readonly solver  = solver;
    static readonly gallery = gallery;
    static readonly decomp  = decomp;
    static readonly EPS     = MatrixBase.EPS;

    // -------- FACTORY Float64M (default) --------
    static zeros(rows: number, cols: number): Matrix<Float64M> {
        return new Matrix<Float64M>(rows, cols);
    }
    static ones(rows: number, cols: number): Matrix<Float64M> {
        const m = new Matrix<Float64M>(rows, cols);
        m.data.fill(Float64M.one);
        return m;
    }
    static identity(n: number): Matrix<Float64M> {
        const m = new Matrix<Float64M>(n, n);
        for (let i = 0; i < n; i++) m.data[i * n + i] = Float64M.one;
        return m;
    }
    static diag(n: number, k: number): Matrix<Float64M> {
        const m = new Matrix<Float64M>(n, n);
        const kv = new Float64M(k);
        for (let i = 0; i < n; i++) m.data[i * n + i] = kv;
        return m;
    }
    static diagFromArray(arr: number[] | Float64Array): Matrix<Float64M> {
        const n = arr.length;
        const m = new Matrix<Float64M>(n, n);
        for (let i = 0; i < n; i++) m.data[i * n + i] = new Float64M(arr[i]);
        return m;
    }
    static fromArray(data: number[][]): Matrix<Float64M> {
        const rows = data.length, cols = data[0]?.length ?? 0;
        const m = new Matrix<Float64M>(rows, cols);
        for (let i = 0; i < rows; i++)
            for (let j = 0; j < cols; j++)
                m.data[i * cols + j] = new Float64M(data[i][j]);
        return m;
    }
    static fromTypedArray<T extends INumeric<T>>(data: T[][], zero: T, one: T): Matrix<T> {
        const rows = data.length, cols = data[0]?.length ?? 0;
        const m = new Matrix<T>(rows, cols, zero, one);
        for (let i = 0; i < rows; i++)
            for (let j = 0; j < cols; j++)
                m.data[i * cols + j] = data[i][j];
        return m;
    }

    // -------- FACTORY tipo generico --------
    static zerosOf<T extends INumeric<T>>(r: number, c: number, zero: T, one: T): Matrix<T> {
        return new Matrix<T>(r, c, zero, one);
    }
    static identityOf<T extends INumeric<T>>(n: number, zero: T, one: T): Matrix<T> {
        const m = new Matrix<T>(n, n, zero, one);
        for (let i = 0; i < n; i++) m.data[i * n + i] = one;
        return m;
    }

    // -------- FACTORY strutturate (Float64M) --------
    static random   = random;
    static sparse   = sparse;
    static toeplitz = toeplitz;
    static vander   = vander;
    static hankel   = hankel;
    static tril     = tril;
    static triu     = triu;
}
