// Matrix.ts
import { MatrixBase }  from "./core/MatrixBase";
import { INumeric, Float64M }  from "./type";

// --- OPS ---
import * as addOps    from "./ops/add";
import * as subOps    from "./ops/subtract";
import * as mulOps    from "./ops/multiply";
import * as dotOps    from "./ops/dotOps";
import * as statOps   from "./ops/statistics";
import * as unaryOps  from "./ops/unary";
import * as transOps  from "./ops/transform";
import * as propOps   from "./ops/hasProperty";
import { equal }      from "./ops/equal";
import { det }        from "./ops/det";
import { norm }       from "./ops/norm";
import { pow }        from "./ops/pow";

// --- GALLERY / SOLVER / DECOMP / INVERSE ---
import * as gallery      from "./init/known";
import * as solver       from "./solver";
import * as decomp       from "./decomposition";
import { smartInverse }  from "./algoritm/inverse";

// --- INIT ---
import { zeros, ones, identity, diag, diagFromArray, zerosLike, identityLike } from "./init/init";
import { hankel, random, sparse, toeplitz, vander } from "./init";
import { tril, triu } from "./decomposition";

// ============================================================
//  Matrix<T>
//
//  Parametro di default = Float64M → backward-compatible con
//  il vecchio codice che usava Matrix come matrice di float.
// ============================================================

export class Matrix<T extends INumeric<T> = Float64M> extends MatrixBase<T> {

    // ----------------------------------------------------------
    // COSTRUTTORE  (backward-compatible)
    //
    //  Nuova API:   new Matrix(rows, cols, zero, one [, data])
    //  Vecchia API: new Matrix(rows, cols)
    //               new Matrix(rows, cols, Float64Array)
    //               new Matrix(rows, cols, number[])
    // ----------------------------------------------------------
    constructor(
        rows: number,
        cols: number,
        zeroOrData?: T | Float64Array | ArrayLike<number> | Array<number>,
        one?: T,
        data?: Array<T>
    ) {
        if (zeroOrData === undefined || zeroOrData === null) {
            // new Matrix(rows, cols) → Float64M zeros
            const f0 = Float64M.zero as unknown as T;
            const f1 = Float64M.one  as unknown as T;
            super(rows, cols, f0, f1);
        } else if (zeroOrData instanceof Float64Array || Array.isArray(zeroOrData)) {
            // Vecchia API: new Matrix(rows, cols, Float64Array | number[])
            const f0 = Float64M.zero as unknown as T;
            const f1 = Float64M.one  as unknown as T;
            const wrapped = Array.from(zeroOrData as ArrayLike<number>)
                .map(v => new Float64M(v) as unknown as T);
            super(rows, cols, f0, f1, wrapped);
        } else {
            // Nuova API: new Matrix(rows, cols, zero, one [, data])
            super(rows, cols, zeroOrData as T, one!, data);
        }
    }

    // ----------------------------------------------------------
    // get / set  — override che accetta anche number per Float64M
    // ----------------------------------------------------------

    /**
     * Restituisce il valore grezzo di tipo T.
     * Per Float64M puoi usare .get(i,j).value  oppure .getNum(i,j).
     */
    get(i: number, j: number): T {
        return this.data[this.idx(i, j)];
    }

    /** Restituisce sempre un number (via toNumber()). Comodo nei test float. */
    getNum(i: number, j: number): number {
        return this.data[this.idx(i, j)].toNumber();
    }

    set(i: number, j: number, v: T | number): void {
        if (typeof v === "number") {
            this.data[this.idx(i, j)] = this.zero.fromNumber(v);
        } else {
            this.data[this.idx(i, j)] = v as T;
        }
    }

    // ----------------------------------------------------------
    // UGUAGLIANZA
    // ----------------------------------------------------------
    equals(B: Matrix<T>, tol: number = MatrixBase.EPS): boolean {
        return equal(this, B, tol);
    }

    // ----------------------------------------------------------
    // ARITMETICA
    // ----------------------------------------------------------
    add(B: Matrix<T> | number): Matrix<T>  { return addOps.add.call(this, B); }
    sub(B: Matrix<T> | number): Matrix<T>  { return subOps.subtract.call(this, B); }
    mul(B: Matrix<T> | number): Matrix<T>  { return mulOps.multiply.call(this, B); }
    pow(exp: number): Matrix<T>            { return pow.call(this, exp); }

    // ----------------------------------------------------------
    // DOT OPS
    // ----------------------------------------------------------
    dotMul(B: Matrix<T> | number): Matrix<T>   { return dotOps.dotMultiply.call(this, B); }
    dotDiv(B: Matrix<T> | number): Matrix<T>   { return dotOps.dotDivide.call(this, B); }
    dotPow(exp: number | Matrix<T>): Matrix<T> { return dotOps.dotPow.call(this, exp); }

    // ----------------------------------------------------------
    // ALGEBRA LINEARE
    // ----------------------------------------------------------
    det(): T                  { return det(this); }
    norm(type?: any): number  { return norm.call(this, type); }
    inv(): Matrix<T>          { return smartInverse(this); }
    inverse(): Matrix<T>      { return this.inv(); }

    trace(): T { return unaryOps.trace.call(this); }
    totalSum(): T { return addOps.totalSum.call(this); }

    // ----------------------------------------------------------
    // STATISTICHE
    // ----------------------------------------------------------
    sum(dim: 1 | 2 = 1): Matrix<T>  { return statOps.sum.call(this, dim); }
    max(dim: 1 | 2 = 1)             { return statOps.max.call(this, dim); }
    min(dim: 1 | 2 = 1)             { return statOps.min.call(this, dim); }
    mean(dim: 1 | 2 = 1): Matrix<T> { return statOps.mean.call(this, dim); }

    // ----------------------------------------------------------
    // TRASFORMAZIONI
    // ----------------------------------------------------------
    t(): Matrix<T>                                                        { return transOps.transpose.call(this); }
    reshape(r: number, c: number): Matrix<T>                              { return transOps.reshape.call(this, r, c); }
    repmat(r: number, c: number): Matrix<T>                               { return transOps.repmat.call(this, r, c); }
    flip(dim: 1 | 2 = 1): Matrix<T>                                       { return transOps.flip.call(this, dim); }
    rot90(k: number = 1): Matrix<T>                                       { return transOps.rot90.call(this, k); }
    slice(rs: number, re: number, cs: number, ce: number): Matrix<T>      { return transOps.slice.call(this, rs, re, cs, ce); }

    // ----------------------------------------------------------
    // UNARIE
    // ----------------------------------------------------------
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

    // ----------------------------------------------------------
    // PROPRIETÀ STRUTTURALI
    // ----------------------------------------------------------
    isSquare(): boolean                        { return propOps.isSquare.call(this); }
    isSymmetric(tol?: number): boolean         { return propOps.isSymmetric.call(this, tol); }
    isUpperTriangular(tol?: number): boolean   { return propOps.isUpperTriangular.call(this, tol); }
    isLowerTriangular(tol?: number): boolean   { return propOps.isLowerTriangular.call(this, tol); }
    isDiagonal(tol?: number): boolean          { return propOps.isDiagonal.call(this, tol); }
    isIdentity(tol?: number): boolean          { return propOps.isIdentity.call(this, tol); }
    isOrthogonal(tol?: number): boolean        { return propOps.isOrthogonal.call(this, tol); }
    isZeroMatrix(tol?: number): boolean        { return propOps.isZeroMatrix.call(this, tol); }
    isInvertible(tol?: number): boolean        { return propOps.isInvertible.call(this, tol); }
    isSingular(tol?: number): boolean          { return propOps.isSingular.call(this, tol); }
    isPositiveDefinite(): boolean              { return propOps.isPositiveDefinite.call(this); }
    isPositiveSemiDefinite(tol?: number): boolean  { return propOps.isPositiveSemiDefinite.call(this, tol); }
    isDiagonallyDominant(): boolean            { return propOps.isDiagonallyDominant.call(this); }
    hasZeroTrace(tol?: number): boolean        { return propOps.hasZeroTrace.call(this, tol); }
    hasFiniteValues(): boolean                 { return propOps.hasFiniteValues.call(this); }
    isStochastic(tol?: number): boolean        { return propOps.isStochastic.call(this, tol); }

    // ----------------------------------------------------------
    // SOLVER
    // ----------------------------------------------------------
    solve(b: Matrix<T>, method: string = "LUP"): Matrix<T> {
        return solver.solve.call(this, b, method);
    }

    // ----------------------------------------------------------
    // FACTORY — tipo generico (zerosOf / identityOf)
    // ----------------------------------------------------------

    /**
     * Crea una Matrix<T> di zeri del tipo T indicato da zero/one.
     *   Matrix.zerosOf(3, 3, Rational.zero, Rational.one)
     *   Matrix.zerosOf(3, 3, Complex.zero, Complex.one)
     */
    static zerosOf<T extends INumeric<T>>(
        rows: number, cols: number, zero: T, one: T
    ): Matrix<T> {
        return new Matrix<T>(rows, cols, zero, one);
    }

    /** Matrice identità del tipo T. */
    static identityOf<T extends INumeric<T>>(n: number, zero: T, one: T): Matrix<T> {
        const m = new Matrix<T>(n, n, zero, one);
        for (let i = 0; i < n; i++) m.set(i, i, one);
        return m;
    }

    // ----------------------------------------------------------
    // FACTORY — Float64M (default, backward-compatible)
    // ----------------------------------------------------------

    static zeros(rows: number, cols: number): Matrix<Float64M> {
        return new Matrix<Float64M>(rows, cols);   // usa il ctor backward-compat
    }

    static ones(rows: number, cols: number): Matrix<Float64M> {
        const m = new Matrix<Float64M>(rows, cols);
        m.data.fill(Float64M.one);
        return m;
    }

    static identity(n: number): Matrix<Float64M> {
        const m = new Matrix<Float64M>(n, n);
        for (let i = 0; i < n; i++) m.set(i, i, Float64M.one);
        return m;
    }

    static diag(n: number, k: number): Matrix<Float64M> {
        const m = new Matrix<Float64M>(n, n);
        const kVal = new Float64M(k);
        for (let i = 0; i < n; i++) m.set(i, i, kVal);
        return m;
    }

    static diagFromArray(arr: Array<Float64M>| number[]): Matrix<Float64M> {
        const n = arr.length;
        const m = new Matrix<Float64M>(n, n);
        for (let i = 0; i < n; i++) m.set(i, i, arr[i]);
        return m;
    }

    /**
     * Costruisce Matrix<Float64M> da array 2D di number.
     *   Matrix.fromArray([[1,2],[3,4]])
     */
    static fromArray(data: number[][]): Matrix<Float64M> {
        const rows = data.length;
        const cols = data[0]?.length ?? 0;
        const m = new Matrix<Float64M>(rows, cols);
        for (let i = 0; i < rows; i++)
            for (let j = 0; j < cols; j++)
                m.setNum(i, j, data[i][j]);
        return m;
    }

    /**
     * Costruisce Matrix<T> da array 2D di valori T.
     *   Matrix.fromTypedArray([[r1,r2],[r3,r4]], Rational.zero, Rational.one)
     */
    static fromTypedArray<T extends INumeric<T>>(
        data: T[][], zero: T, one: T
    ): Matrix<T> {
        const rows = data.length;
        const cols = data[0]?.length ?? 0;
        const m = new Matrix<T>(rows, cols, zero, one);
        for (let i = 0; i < rows; i++)
            for (let j = 0; j < cols; j++)
                m.set(i, j, data[i][j]);
        return m;
    }

    // ----------------------------------------------------------
    // STATIC — matrici strutturate e namespace
    // ----------------------------------------------------------
    static random    = random;
    static sparse    = sparse;
    static toeplitz  = toeplitz;
    static vander    = vander;
    static hankel    = hankel;
    static tril      = tril;
    static triu      = triu;

    static readonly solver  = solver;
    static readonly gallery = gallery;
    static readonly decomp  = decomp;
    static readonly EPS     = MatrixBase.EPS;
}
