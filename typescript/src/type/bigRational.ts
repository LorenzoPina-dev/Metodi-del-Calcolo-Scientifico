// type/bigRational.ts
import { INumeric } from "./interface";

export class Rational implements INumeric<Rational> {
    public readonly num: bigint;
    public readonly den: bigint;

    static readonly zero: Rational = new Rational(0);
    static readonly one: Rational  = new Rational(1);

    constructor(num: bigint | number, den: bigint | number = 1) {
        let n = BigInt(typeof num === "number" ? Math.trunc(num) : num);
        let d = BigInt(typeof den === "number" ? Math.trunc(den) : den);

        if (d === BigInt(0)) throw new Error("Rational: denominatore zero.");
        if (d < BigInt(0)) { n = -n; d = -d; }

        const g = Rational.gcd(n < BigInt(0) ? -n : n, d);
        this.num = n / g;
        this.den = d / g;
    }

    // ---- GCD ----
    private static gcd(a: bigint, b: bigint): bigint {
        while (b > BigInt(0)) { [a, b] = [b, a % b]; }
        return a;
    }

    // ---- Factory ----
    fromNumber(n: number): Rational {
        if (!Number.isFinite(n)) throw new Error("Rational.fromNumber: valore non finito.");
        if (Number.isInteger(n)) return new Rational(BigInt(n));
        // Approssima con denominatore 10^15 per mantenere la precisione float
        const SCALE = 1_000_000_000_000_000;
        const rounded = BigInt(Math.round(n * 1e15));
        return new Rational(rounded, SCALE);
    }

    toNumber(): number {
        return Number(this.num) / Number(this.den);
    }

    // ---- Aritmetica ----
    negate(): Rational {
        return new Rational(-this.num, this.den);
    }
    add(other: Rational): Rational {
        return new Rational(
            this.num * other.den + other.num * this.den,
            this.den * other.den
        );
    }
    subtract(other: Rational): Rational {
        return new Rational(
            this.num * other.den - other.num * this.den,
            this.den * other.den
        );
    }
    multiply(other: Rational): Rational {
        return new Rational(this.num * other.num, this.den * other.den);
    }
    divide(other: Rational): Rational {
        if (other.num === BigInt(0)) throw new Error("Rational: divisione per zero!");
        return new Rational(this.num * other.den, this.den * other.num);
    }

    // ---- Funzioni ----
    abs(): Rational {
        return this.num < BigInt(0) ? new Rational(-this.num, this.den) : this;
    }
    sqrt(): Rational {
        // Approssima: restituisce p/q tale che (p/q)^2 ≈ this
        return this.fromNumber(Math.sqrt(this.toNumber()));
    }
    round(): Rational {
        // Arrotonda all'intero più vicino
        const isNeg = this.num < BigInt(0);
        const absNum = isNeg ? -this.num : this.num;
        const half = this.den / BigInt(2);
        const rounded = (absNum + half) / this.den;
        return new Rational(isNeg ? -rounded : rounded);
    }

    // ---- Comparazione ----
    equals(other: Rational): boolean {
        return this.num === other.num && this.den === other.den;
    }
    greaterThan(other: Rational): boolean {
        return this.num * other.den > other.num * this.den;
    }
    lessThan(other: Rational): boolean {
        return this.num * other.den < other.num * this.den;
    }
    /** Per Rational la tolleranza è ignorata: zero esatto. */
    isNearZero(_tol: number): boolean {
        return this.num === BigInt(0);
    }

    // ---- Output ----
    toString(): string {
        return this.den === BigInt(1) ? `${this.num}` : `${this.num}/${this.den}`;
    }

    toDecimal(precision: number = 10): string {
        const factor = BigInt(10 ** precision);
        const value = (this.num * factor) / this.den;
        const s = value.toString().padStart(precision + 1, "0");
        const dotIndex = s.length - precision;
        return s.slice(0, dotIndex) + "." + s.slice(dotIndex);
    }
}