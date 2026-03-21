// type/float.ts
import { INumeric } from "./interface";

export class Float64M implements INumeric<Float64M> {
    constructor(public readonly value: number) {}

    static readonly zero: Float64M = new Float64M(0);
    static readonly one: Float64M  = new Float64M(1);

    // ---- Factory ----
    fromNumber(n: number): Float64M { return new Float64M(n); }
    toNumber(): number { return this.value; }

    /**
     * valueOf() permette alla coercizione JS di funzionare:
     *   A.get(i,j) + 3          → number
     *   totalSum * 2            → number
     *   Math.abs(get(i,j))      → number
     *   toBeCloseTo(get(i,j))   → confronto numerico corretto
     *
     * NOTA: toBe() usa Object.is() (strict), quindi
     *   expect(get(i,j)).toBe(5) fallisce ancora.
     *   Usare expect(get(i,j).value).toBe(5)
     *   oppure  expect(get(i,j).toNumber()).toBe(5).
     */
    valueOf(): number { return this.value; }

    // ---- Aritmetica ----
    negate(): Float64M    { return new Float64M(-this.value); }
    add(other: Float64M): Float64M      { return new Float64M(this.value + other.value); }
    subtract(other: Float64M): Float64M { return new Float64M(this.value - other.value); }
    multiply(other: Float64M): Float64M { return new Float64M(this.value * other.value); }
    divide(other: Float64M): Float64M {
        if (other.value === 0) throw new Error("Float64M: divisione per zero!");
        return new Float64M(this.value / other.value);
    }

    // ---- Funzioni ----
    abs(): Float64M   { return new Float64M(Math.abs(this.value)); }
    sqrt(): Float64M  { return new Float64M(Math.sqrt(this.value)); }
    round(): Float64M { return new Float64M(Math.round(this.value)); }

    // ---- Comparazione ----
    greaterThan(other: Float64M): boolean { return this.value > other.value; }
    lessThan(other: Float64M): boolean    { return this.value < other.value; }
    equals(other: Float64M): boolean      { return Math.abs(this.value - other.value) < Number.EPSILON; }
    isNearZero(tol: number): boolean      { return Math.abs(this.value) < tol; }

    toString(): string { return this.value.toString(); }
}
