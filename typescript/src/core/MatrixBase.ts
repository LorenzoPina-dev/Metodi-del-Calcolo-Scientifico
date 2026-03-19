// core/MatrixBase.ts

export class MatrixBase {
    data: Float64Array;
    rows: number;
    cols: number;

    constructor(rows: number, cols: number, data?: Float64Array) {
        this.rows = rows;
        this.cols = cols;
        this.data = data ?? new Float64Array(rows * cols);
    }

    // ---------------- INDEX ----------------
    protected idx(i: number, j: number): number {
        return i * this.cols + j;
    }

    get(i: number, j: number): number {
        return this.data[this.idx(i, j)];
    }

    set(i: number, j: number, v: number): void {
        this.data[this.idx(i, j)] = v;
    }

    // ---------------- CONSTANTS ----------------
    static EPS = 1e-10;

    static isZero(x: number): boolean {
        return Math.abs(x) < MatrixBase.EPS;
    }

    // ---------------- COPY ----------------
    clone(): this {
        const Ctor = this.constructor as any;
        return new Ctor(
            this.rows,
            this.cols,
            new Float64Array(this.data)
        );
    }

    // ---------------- STAMPA ----------------
    toString(): string {
        let s = "";

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                s += this.get(i, j).toFixed(3) + " ";
            }
            s += "\n";
        }

        return s;
    }
}