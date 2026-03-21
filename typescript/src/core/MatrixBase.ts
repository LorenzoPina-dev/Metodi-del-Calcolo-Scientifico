// core/MatrixBase.ts

import { INumeric } from "../type";

export class MatrixBase< T extends INumeric<T> > {
    data: Array<T>;
    rows: number;
    cols: number;

    constructor(rows: number, cols: number, data?: Array<T>) {
        this.rows = rows;
        this.cols = cols;
        this.data = data ?? new Array<T>(rows * cols);
    }

    // ---------------- INDEX ----------------
    protected idx(i: number, j: number): number {
        return i * this.cols + j;
    }

    get(i: number, j: number): T {
        return this.data[this.idx(i, j)];
    }

    set(i: number, j: number, v: T): void {
        this.data[this.idx(i, j)] = v;
    }

    // ---------------- CONSTANTS ----------------
    static EPS = 1e-10;

    static isZero(x: INumeric<any>): boolean {
        return x.abs().lessThan(MatrixBase.EPS);
    }


    // ---------------- COPY ----------------
    clone(): this {
        const Ctor = this.constructor as any;
        return new Ctor(
            this.rows,
            this.cols,
            Array.from(this.data)
        );
    }

    // ---------------- STAMPA ----------------
    toString(): string {
        let s = "";

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                s += this.get(i, j).toString() + " ";
            }
            s += "\n";
        }

        return s;
    }
}