// type/bigRational.ts
import { INumeric } from "./interface";

export class Rational implements INumeric<Rational> {
    readonly kind = "rational" as const;

    public readonly num: bigint;
    public readonly den: bigint;

    static readonly zero: Rational = new Rational(0n, 1n, true);
    static readonly one:  Rational = new Rational(1n, 1n, true);

    constructor(num: bigint | number, den: bigint | number = 1n, _reduced = false) {
        let n = typeof num === "bigint" ? num : BigInt(Math.trunc(num as number));
        let d = typeof den === "bigint" ? den : BigInt(Math.trunc(den as number));

        if (d === 0n) throw new Error("Rational: denominatore zero.");
        if (!_reduced) {
            if (d < 0n) { n = -n; d = -d; }
            const g = Rational.gcd(n < 0n ? -n : n, d);
            if (g > 1n) { n = n / g; d = d / g; }
        }
        this.num = n;
        this.den = d;
    }

    // ---- GCD binario di Lehmer ----
    static gcd(a: bigint, b: bigint): bigint {
        if (a === 0n) return b;
        if (b === 0n) return a;
        let shift = 0n;
        while (((a | b) & 1n) === 0n) { a >>= 1n; b >>= 1n; shift++; }
        while ((a & 1n) === 0n) a >>= 1n;
        while (b !== 0n) {
            while ((b & 1n) === 0n) b >>= 1n;
            if (a > b) { const t = a; a = b; b = t; }
            b -= a;
        }
        return a << shift;
    }

    private static reduced(n: bigint, d: bigint): Rational {
        return new Rational(n, d, true);
    }

    // ---- Factory ----
    fromNumber(n: number): Rational {
        if (!Number.isFinite(n)) throw new Error("Rational.fromNumber: valore non finito.");
        if (Number.isInteger(n)) return Rational.reduced(BigInt(n), 1n);
        const SCALE = 1_000_000_000_000_000n;
        const rounded = BigInt(Math.round(n * 1e15));
        const g = Rational.gcd(rounded < 0n ? -rounded : rounded, SCALE);
        return Rational.reduced(rounded / g, SCALE / g);
    }

    toNumber(): number { return Number(this.num) / Number(this.den); }

    // ---- Aritmetica con cross-cancellation ----

    negate(): Rational { return Rational.reduced(-this.num, this.den); }

    add(o: Rational): Rational {
        if (this.den === o.den) {
            const n = this.num + o.num;
            if (n === 0n) return Rational.zero;
            const g = Rational.gcd(n < 0n ? -n : n, this.den);
            return Rational.reduced(n / g, this.den / g);
        }
        const g   = Rational.gcd(this.den, o.den);
        const den = this.den / g * o.den;
        const n   = this.num * (o.den / g) + o.num * (this.den / g);
        if (n === 0n) return Rational.zero;
        const g2 = Rational.gcd(n < 0n ? -n : n, g);
        return Rational.reduced(n / g2, den / g2);
    }

    subtract(o: Rational): Rational {
        if (this.den === o.den) {
            const n = this.num - o.num;
            if (n === 0n) return Rational.zero;
            const g = Rational.gcd(n < 0n ? -n : n, this.den);
            return Rational.reduced(n / g, this.den / g);
        }
        const g   = Rational.gcd(this.den, o.den);
        const den = this.den / g * o.den;
        const n   = this.num * (o.den / g) - o.num * (this.den / g);
        if (n === 0n) return Rational.zero;
        const g2 = Rational.gcd(n < 0n ? -n : n, g);
        return Rational.reduced(n / g2, den / g2);
    }

    multiply(o: Rational): Rational {
        if (this.num === 0n || o.num === 0n) return Rational.zero;
        const g1 = Rational.gcd(this.num < 0n ? -this.num : this.num, o.den);
        const g2 = Rational.gcd(o.num  < 0n ? -o.num  : o.num,  this.den);
        return Rational.reduced(
            (this.num / g1) * (o.num / g2),
            (this.den / g2) * (o.den / g1)
        );
    }

    divide(o: Rational): Rational {
        if (o.num === 0n) throw new Error("Rational: divisione per zero!");
        const g1 = Rational.gcd(this.num < 0n ? -this.num : this.num, o.num < 0n ? -o.num : o.num);
        const g2 = Rational.gcd(this.den, o.den);
        let n = (this.num / g1) * (o.den / g2);
        let d = (this.den / g2) * (o.num / g1);
        if (d < 0n) { n = -n; d = -d; }
        return Rational.reduced(n, d);
    }

    // ---- Funzioni ----
    abs(): Rational {
        return this.num < 0n ? Rational.reduced(-this.num, this.den) : this;
    }
    sqrt(): Rational { return this.fromNumber(Math.sqrt(this.toNumber())); }
    round(): Rational {
        const absNum  = this.num < 0n ? -this.num : this.num;
        const rounded = (absNum + this.den / 2n) / this.den;
        return Rational.reduced(this.num < 0n ? -rounded : rounded, 1n);
    }
    /** I razionali sono auto-coniugati (sono reali). */
    conjugate(): Rational { return this; }

    // ---- Comparazione ----
    equals(o: Rational): boolean      { return this.num === o.num && this.den === o.den; }
    greaterThan(o: Rational): boolean { return this.num * o.den > o.num * this.den; }
    lessThan(o: Rational): boolean    { return this.num * o.den < o.num * this.den; }
    isNearZero(_tol: number): boolean { return this.num === 0n; }

    // ---- Output ----
    toString(): string {
        return this.den === 1n ? `${this.num}` : `${this.num}/${this.den}`;
    }
    toDecimal(precision = 10): string {
        const factor = 10n ** BigInt(precision);
        const value  = (this.num * factor) / this.den;
        const s      = value.toString().padStart(precision + 1, "0");
        const dot    = s.length - precision;
        return s.slice(0, dot) + "." + s.slice(dot);
    }
}
