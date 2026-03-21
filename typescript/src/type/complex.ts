// type/complex.ts
import { INumeric } from "./interface";

export class Complex implements INumeric<Complex> {
    readonly kind = "complex" as const;

    constructor(
        public readonly real: number,
        public readonly imag: number
    ) {}

    static readonly zero: Complex = new Complex(0, 0);
    static readonly one:  Complex = new Complex(1, 0);

    // ---- Magnitudine con memoization lazy ----
    private _magSq: number | undefined;
    private _mag:   number | undefined;

    get magnitudeSq(): number {
        if (this._magSq === undefined)
            this._magSq = this.real * this.real + this.imag * this.imag;
        return this._magSq;
    }

    get magnitude(): number {
        if (this._mag === undefined) this._mag = Math.sqrt(this.magnitudeSq);
        return this._mag;
    }

    // ---- Factory ----
    fromNumber(n: number): Complex { return new Complex(n, 0); }
    toNumber(): number             { return this.magnitude; }

    // ---- Aritmetica ----
    negate(): Complex { return new Complex(-this.real, -this.imag); }

    add(o: Complex): Complex {
        return new Complex(this.real + o.real, this.imag + o.imag);
    }
    subtract(o: Complex): Complex {
        return new Complex(this.real - o.real, this.imag - o.imag);
    }
    multiply(o: Complex): Complex {
        return new Complex(
            this.real * o.real - this.imag * o.imag,
            this.real * o.imag + this.imag * o.real
        );
    }
    divide(o: Complex): Complex {
        const denom = o.magnitudeSq;
        if (denom === 0) throw new Error("Complex: divisione per zero!");
        return new Complex(
            (this.real * o.real + this.imag * o.imag) / denom,
            (this.imag * o.real - this.real * o.imag) / denom
        );
    }

    // ---- Funzioni ----
    abs(): Complex { return new Complex(this.magnitude, 0); }
    sqrt(): Complex {
        const r     = this.magnitude;
        const theta = Math.atan2(this.imag, this.real);
        return new Complex(
            Math.sqrt(r) * Math.cos(theta * 0.5),
            Math.sqrt(r) * Math.sin(theta * 0.5)
        );
    }
    round(): Complex { return new Complex(Math.round(this.real), Math.round(this.imag)); }

    /** Coniugato complesso: a+bi → a-bi. */
    conjugate(): Complex { return new Complex(this.real, -this.imag); }

    // ---- Comparazione (su modulo² — evita sqrt inutili) ----
    greaterThan(o: Complex): boolean { return this.magnitudeSq > o.magnitudeSq; }
    lessThan(o: Complex): boolean    { return this.magnitudeSq < o.magnitudeSq; }
    equals(o: Complex): boolean      { return this.real === o.real && this.imag === o.imag; }
    isNearZero(tol: number): boolean { return this.magnitudeSq < tol * tol; }

    toString(): string {
        if (this.imag === 0) return `${this.real}`;
        const sign = this.imag >= 0 ? "+" : "-";
        return `${this.real} ${sign} ${Math.abs(this.imag)}i`;
    }
}
