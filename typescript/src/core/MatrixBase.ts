// core/MatrixBase.ts

import { Float64M, INumeric } from "../type";

export class MatrixBase<T extends INumeric<T>> {
    data: Array<T>;
    rows: number;
    cols: number;

    /** Elemento neutro additivo (0) del tipo T. */
    readonly zero: T;
    /** Elemento neutro moltiplicativo (1) del tipo T. */
    readonly one: T;

    constructor(rows: number, cols: number, zero: T, one: T, data?: Array<T>) {
        this.rows = rows;
        this.cols = cols;
        this.zero = zero;
        this.one  = one;
        this.data = data ?? new Array<T>(rows * cols).fill(zero);
    }

    // ==================== INDICIZZAZIONE ====================

    protected idx(i: number, j: number): number {
        return i * this.cols + j;
    }

    get(i: number, j: number): T {
        return this.data[this.idx(i, j)];
    }

    set(i: number, j: number, v: T): void {
        this.data[this.idx(i, j)] = v;
    }

    /**
     * Imposta un elemento convertendo un number JS tramite fromNumber().
     * Funziona per qualsiasi T che implementi INumeric.
     */
    setNum(i: number, j: number, n: number): void {
        this.data[this.idx(i, j)] = this.zero.fromNumber(n);
    }

    // ==================== ZERO-CHECK ====================

    static readonly EPS = 1e-10;

    isZero(x: T): boolean {
        return x.isNearZero(MatrixBase.EPS);
    }

    // ==================== FACTORY (stesso tipo, stessa sottoclasse) ====================

    /**
     * Crea una nuova matrice zero delle dimensioni richieste,
     * dello stesso tipo concreto (Matrix, ComplexMatrix, …).
     *
     * IMPORTANTE: la sottoclasse DEVE avere un costruttore con firma
     *   (rows, cols, zero, one, data?)
     * come fa Matrix<T>.
     */
    like(rows: number, cols: number): this {
        const Ctor = this.constructor as new (r: number, c: number, z: T, o: T) => this;
        return new Ctor(rows, cols, this.zero, this.one);
    }

    /** Matrice identità n×n dello stesso tipo concreto. */
    likeIdentity(n: number): this {
        const m = this.like(n, n);
        for (let i = 0; i < n; i++) m.set(i, i, this.one);
        return m;
    }

    /** Matrice di soli `one` dello stesso tipo concreto. */
    likeOnes(rows: number, cols: number): this {
        const m = this.like(rows, cols);
        m.data.fill(this.one);
        return m;
    }

    // ==================== COPIA ====================

    clone(): this {
        const Ctor = this.constructor as new (r: number, c: number, z: T, o: T, d: Array<T>) => this;
        return new Ctor(this.rows, this.cols, this.zero, this.one, [...this.data]);
    }

    // ==================== STAMPA ====================

    toString(): string {
        let s = "";
        for (let i = 0; i < this.rows; i++) {
            const row: string[] = [];
            for (let j = 0; j < this.cols; j++) row.push(this.get(i, j).toString());
            s += row.join("\t") + "\n";
        }
        return s;
    }
}
