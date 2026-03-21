// core/MatrixBase.ts
import { INumeric } from "../type";

export class MatrixBase<T extends INumeric<T>> {
    data: Array<T>;
    rows: number;
    cols: number;

    readonly zero: T;
    readonly one:  T;

    /**
     * True se T = Float64M.
     * Attiva i fast-path con Float64Array nelle operazioni critiche.
     */
    readonly isFloat64: boolean;

    constructor(rows: number, cols: number, zero: T, one: T, data?: Array<T>) {
        this.rows      = rows;
        this.cols      = cols;
        this.zero      = zero;
        this.one       = one;
        this.isFloat64 = zero.kind === "float64";
        this.data      = data ?? new Array<T>(rows * cols).fill(zero);
    }

    // ==================== INDICIZZAZIONE ====================

    protected idx(i: number, j: number): number { return i * this.cols + j; }

    get(i: number, j: number): T          { return this.data[i * this.cols + j]; }
    set(i: number, j: number, v: T): void { this.data[i * this.cols + j] = v; }
    setNum(i: number, j: number, n: number): void {
        this.data[i * this.cols + j] = this.zero.fromNumber(n);
    }

    // ==================== ZERO-CHECK ====================

    static readonly EPS = 1e-10;

    isZero(x: T): boolean { return x.isNearZero(MatrixBase.EPS); }

    // ==================== FACTORY ====================

    like(rows: number, cols: number): this {
        const Ctor = this.constructor as new (r: number, c: number, z: T, o: T) => this;
        return new Ctor(rows, cols, this.zero, this.one);
    }

    likeWithData(rows: number, cols: number, data: Array<T>): this {
        const Ctor = this.constructor as new (r: number, c: number, z: T, o: T, d: Array<T>) => this;
        return new Ctor(rows, cols, this.zero, this.one, data);
    }

    likeIdentity(n: number): this {
        const m = this.like(n, n);
        for (let i = 0; i < n; i++) m.data[i * n + i] = this.one;
        return m;
    }

    likeOnes(rows: number, cols: number): this {
        const m = this.like(rows, cols);
        m.data.fill(this.one);
        return m;
    }

    // ==================== COPIA ====================

    clone(): this {
        const Ctor = this.constructor as new (r: number, c: number, z: T, o: T, d: Array<T>) => this;
        return new Ctor(this.rows, this.cols, this.zero, this.one, this.data.slice());
    }

    // ==================== TRASPOSTA CONIUGATA (Hermitian adjoint) ====================

    /**
     * Trasposta coniugata A^H: (A^H)_{ij} = conj(A_{ji}).
     *
     * Per tipi reali (Float64M, Rational) coincide con la semplice trasposta
     * perché `conjugate()` restituisce `this`.
     * Per Complex è la vera trasposta coniugata usata nel QR e nel solver.
     */
    ct(): this {
        const R = this.rows, C = this.cols;
        const ad = this.data;
        const outData = new Array<T>(R * C);
        for (let i = 0; i < R; i++) {
            const iOff = i * C;
            for (let j = 0; j < C; j++) {
                outData[j * R + i] = ad[iOff + j].conjugate();
            }
        }
        return this.likeWithData(C, R, outData);
    }

    // ==================== STAMPA ====================

    toString(): string {
        const R = this.rows, C = this.cols;
        let s = "";
        for (let i = 0; i < R; i++) {
            const off = i * C;
            for (let j = 0; j < C; j++) {
                if (j > 0) s += "\t";
                s += this.data[off + j].toString();
            }
            s += "\n";
        }
        return s;
    }
}
