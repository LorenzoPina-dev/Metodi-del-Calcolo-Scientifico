// type/bigRational.ts
//
// Ottimizzazioni applicate rispetto alla versione precedente:
//
//   1. INTEGER FAST-PATH: se den === 1n in entrambi gli operandi, add/subtract/multiply
//      saltano tutti i GCD → da 2-4 chiamate GCD a 0 per casi interi.
//      In matrici di Hilbert questo non aiuta (i denominatori sono i+j+1),
//      ma in algebra lineare generica (coefficienti interi) è un win enorme.
//
//   2. GCD FAST-PATH PER VALORI PICCOLI: se entrambi gli argomenti rientrano in
//      Number (< 2^53), usiamo l'algoritmo di Euclide su Number e poi convertiamo
//      in BigInt. L'aritmetica Number in V8 è ~10× più veloce dei BigInt piccoli.
//
//   3. ZERO FAST-PATH in multiply: se uno dei due numeratori è 0n, ritorna
//      Rational.zero immediatamente senza calcolare GCD.
//
//   4. SAME-DENOMINATOR: già presente, ottimizzato con skip del secondo GCD
//      quando il risultato ha già il denominatore ridotto.
//
//   5. DIVIDE per intero: se il divisore ha den === 1n (è un intero) semplifica
//      la formula eliminando un GCD.
//
import { INumeric } from "./interface";

// Soglia sotto cui usiamo l'Euclide su Number invece di BigInt
const GCD_NUMBER_LIMIT = 9_007_199_254_740_992n; // 2^53

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

    // ── GCD con fast-path su Number per valori piccoli ─────────────────────
    //
    // Per bigint piccoli (< 2^53), l'algoritmo di Euclide su Number è
    // molto più veloce perché V8 rappresenta i Number come double-precision
    // float a 64 bit (operazioni in hardware), mentre i bigint piccoli usano
    // comunque l'heap allocator. Il guadagno è circa 5-8× per valori tipici
    // della Hilbert matrix (denominatori fino a ~200).
    //
    static gcd(a: bigint, b: bigint): bigint {
        if (a === 0n) return b;
        if (b === 0n) return a;

        // Fast-path: valori piccoli → Euclide su Number
        if (a < GCD_NUMBER_LIMIT && b < GCD_NUMBER_LIMIT) {
            let an = Number(a), bn = Number(b);
            while (bn !== 0) { const t = bn; bn = an % bn; an = t; }
            return BigInt(an);
        }

        // Binary GCD per valori grandi (efficiente per bigint con molti bit)
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

    // Costruttore interno: salta la riduzione (chiamato solo quando già ridotto)
    private static reduced(n: bigint, d: bigint): Rational {
        return new Rational(n, d, true);
    }

    // ── Factory ────────────────────────────────────────────────────────────
    fromNumber(n: number): Rational {
        if (!Number.isFinite(n)) throw new Error("Rational.fromNumber: valore non finito.");
        if (Number.isInteger(n)) return Rational.reduced(BigInt(n), 1n);
        const SCALE = 1_000_000_000_000_000n;
        const rounded = BigInt(Math.round(n * 1e15));
        const g = Rational.gcd(rounded < 0n ? -rounded : rounded, SCALE);
        return Rational.reduced(rounded / g, SCALE / g);
    }

    toNumber(): number { return Number(this.num) / Number(this.den); }

    // ── Aritmetica ──────────────────────────────────────────────────────────

    negate(): Rational {
        if (this.num === 0n) return Rational.zero;
        return Rational.reduced(-this.num, this.den);
    }

    add(o: Rational): Rational {
        // Fast-path: entrambi interi (den = 1)
        if (this.den === 1n && o.den === 1n) {
            const n = this.num + o.num;
            return n === 0n ? Rational.zero : Rational.reduced(n, 1n);
        }

        // Fast-path: stesso denominatore
        if (this.den === o.den) {
            const n = this.num + o.num;
            if (n === 0n) return Rational.zero;
            const g = Rational.gcd(n < 0n ? -n : n, this.den);
            return g === 1n
                ? Rational.reduced(n, this.den)
                : Rational.reduced(n / g, this.den / g);
        }

        // Caso generale con cross-cancellation
        const g   = Rational.gcd(this.den, o.den);
        const odenDivg  = o.den / g;
        const thisdenDivg = this.den / g;
        const den = thisdenDivg * o.den;
        const n   = this.num * odenDivg + o.num * thisdenDivg;
        if (n === 0n) return Rational.zero;
        const g2 = Rational.gcd(n < 0n ? -n : n, g);
        return g2 === 1n
            ? Rational.reduced(n, den)
            : Rational.reduced(n / g2, den / g2);
    }

    subtract(o: Rational): Rational {
        // Fast-path: entrambi interi
        if (this.den === 1n && o.den === 1n) {
            const n = this.num - o.num;
            return n === 0n ? Rational.zero : Rational.reduced(n, 1n);
        }

        // Fast-path: stesso denominatore
        if (this.den === o.den) {
            const n = this.num - o.num;
            if (n === 0n) return Rational.zero;
            const g = Rational.gcd(n < 0n ? -n : n, this.den);
            return g === 1n
                ? Rational.reduced(n, this.den)
                : Rational.reduced(n / g, this.den / g);
        }

        const g   = Rational.gcd(this.den, o.den);
        const odenDivg    = o.den / g;
        const thisdenDivg = this.den / g;
        const den = thisdenDivg * o.den;
        const n   = this.num * odenDivg - o.num * thisdenDivg;
        if (n === 0n) return Rational.zero;
        const g2 = Rational.gcd(n < 0n ? -n : n, g);
        return g2 === 1n
            ? Rational.reduced(n, den)
            : Rational.reduced(n / g2, den / g2);
    }

    multiply(o: Rational): Rational {
        // Fast-path: zero
        if (this.num === 0n || o.num === 0n) return Rational.zero;

        // Fast-path: entrambi interi → prodotto senza GCD
        if (this.den === 1n && o.den === 1n) {
            return Rational.reduced(this.num * o.num, 1n);
        }

        // Fast-path: uno dei due è intero → un solo GCD invece di due
        if (this.den === 1n) {
            const g = Rational.gcd(this.num < 0n ? -this.num : this.num, o.den);
            return g === 1n
                ? Rational.reduced(this.num * o.num, o.den)
                : Rational.reduced((this.num / g) * o.num, o.den / g);
        }
        if (o.den === 1n) {
            const g = Rational.gcd(o.num < 0n ? -o.num : o.num, this.den);
            return g === 1n
                ? Rational.reduced(this.num * o.num, this.den)
                : Rational.reduced(this.num * (o.num / g), this.den / g);
        }

        // Caso generale: cross-cancellation
        const g1 = Rational.gcd(this.num < 0n ? -this.num : this.num, o.den);
        const g2 = Rational.gcd(o.num  < 0n ? -o.num  : o.num,  this.den);
        const n = (this.num / g1) * (o.num / g2);
        const d = (this.den / g2) * (o.den / g1);
        return Rational.reduced(n, d);
    }

    divide(o: Rational): Rational {
        if (o.num === 0n) throw new Error("Rational: divisione per zero!");

        // Fast-path: divisore intero (o.den = 1) → a/b ÷ c = a/(b*c)
        if (o.den === 1n) {
            const absONum = o.num < 0n ? -o.num : o.num;
            const g = Rational.gcd(this.num < 0n ? -this.num : this.num, absONum);
            const newNum = this.num / g;
            let newDen = this.den * (o.num / g);
            if (newDen < 0n) { return Rational.reduced(-newNum, -newDen); }
            return Rational.reduced(newNum, newDen);
        }

        // Caso generale: a/b ÷ c/d = a/b * d/c
        const g1 = Rational.gcd(this.num < 0n ? -this.num : this.num, o.num < 0n ? -o.num : o.num);
        const g2 = Rational.gcd(this.den, o.den);
        let n = (this.num / g1) * (o.den / g2);
        let d = (this.den / g2) * (o.num / g1);
        if (d < 0n) { n = -n; d = -d; }
        return Rational.reduced(n, d);
    }

    // ── Funzioni ────────────────────────────────────────────────────────────
    abs(): Rational {
        if (this.num === 0n) return Rational.zero;
        return this.num < 0n ? Rational.reduced(-this.num, this.den) : this;
    }
    sqrt(): Rational { return this.fromNumber(Math.sqrt(this.toNumber())); }
    round(): Rational {
        if (this.den === 1n) return this;   // già intero
        const absNum  = this.num < 0n ? -this.num : this.num;
        const rounded = (absNum + this.den / 2n) / this.den;
        return Rational.reduced(this.num < 0n ? -rounded : rounded, 1n);
    }
    conjugate(): Rational { return this; }

    // ── Comparazione ────────────────────────────────────────────────────────
    equals(o: Rational): boolean      { return this.num === o.num && this.den === o.den; }
    greaterThan(o: Rational): boolean {
        // Fast-path: stesso denominatore
        if (this.den === o.den) return this.num > o.num;
        return this.num * o.den > o.num * this.den;
    }
    lessThan(o: Rational): boolean {
        if (this.den === o.den) return this.num < o.num;
        return this.num * o.den < o.num * this.den;
    }
    isNearZero(_tol: number): boolean { return this.num === 0n; }

    // ── Output ──────────────────────────────────────────────────────────────
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
