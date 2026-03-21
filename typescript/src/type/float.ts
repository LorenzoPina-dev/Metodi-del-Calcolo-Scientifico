// type/float.ts
import { INumeric } from "./interface";

export class Float64M implements INumeric<Float64M> {
    readonly kind = "float64" as const;

    constructor(public readonly value: number) {}

    static readonly zero: Float64M = new Float64M(0);
    static readonly one:  Float64M = new Float64M(1);

    // ---- Factory ----
    fromNumber(n: number): Float64M { return new Float64M(n); }
    toNumber(): number              { return this.value; }
    valueOf(): number               { return this.value; }

    // ---- Aritmetica ----
    negate(): Float64M                  { return new Float64M(-this.value); }
    add(o: Float64M): Float64M          { return new Float64M(this.value + o.value); }
    subtract(o: Float64M): Float64M     { return new Float64M(this.value - o.value); }
    multiply(o: Float64M): Float64M     { return new Float64M(this.value * o.value); }
    divide(o: Float64M): Float64M {
        if (o.value === 0) throw new Error("Float64M: divisione per zero!");
        return new Float64M(this.value / o.value);
    }

    // ---- Funzioni ----
    abs(): Float64M       { return new Float64M(Math.abs(this.value)); }
    sqrt(): Float64M      { return new Float64M(Math.sqrt(this.value)); }
    round(): Float64M     { return new Float64M(Math.round(this.value)); }
    /** I numeri reali sono auto-coniugati. */
    conjugate(): Float64M { return this; }

    // ---- Comparazione ----
    greaterThan(o: Float64M): boolean { return this.value > o.value; }
    lessThan(o: Float64M): boolean    { return this.value < o.value; }
    equals(o: Float64M): boolean      { return Math.abs(this.value - o.value) < Number.EPSILON; }
    isNearZero(tol: number): boolean  { return Math.abs(this.value) < tol; }

    toString(): string { return this.value.toString(); }
}
