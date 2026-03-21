// type/complex.ts
import { INumeric } from "./interface";

export class Complex implements INumeric<Complex> {
    constructor(
        public readonly real: number,
        public readonly imag: number
    ) {}

    static readonly zero: Complex = new Complex(0, 0);
    static readonly one: Complex  = new Complex(1, 0);

    // ---- Magnitudine ----
    get magnitude(): number { return Math.sqrt(this.real ** 2 + this.imag ** 2); }

    // ---- Factory ----
    fromNumber(n: number): Complex { return new Complex(n, 0); }
    /** toNumber restituisce il modulo (usato per norme e confronti scalari). */
    toNumber(): number { return this.magnitude; }

    // ---- Aritmetica ----
    negate(): Complex { return new Complex(-this.real, -this.imag); }

    add(other: Complex): Complex {
        return new Complex(this.real + other.real, this.imag + other.imag);
    }
    subtract(other: Complex): Complex {
        return new Complex(this.real - other.real, this.imag - other.imag);
    }
    multiply(other: Complex): Complex {
        return new Complex(
            this.real * other.real - this.imag * other.imag,
            this.real * other.imag + this.imag * other.real
        );
    }
    divide(other: Complex): Complex {
        const denom = other.real ** 2 + other.imag ** 2;
        if (denom === 0) throw new Error("Complex: divisione per zero!");
        return new Complex(
            (this.real * other.real + this.imag * other.imag) / denom,
            (this.imag * other.real - this.real * other.imag) / denom
        );
    }

    // ---- Funzioni ----
    abs(): Complex { return new Complex(this.magnitude, 0); }
    sqrt(): Complex {
        const r = this.magnitude;
        const theta = Math.atan2(this.imag, this.real);
        return new Complex(
            Math.sqrt(r) * Math.cos(theta / 2),
            Math.sqrt(r) * Math.sin(theta / 2)
        );
    }
    round(): Complex {
        return new Complex(Math.round(this.real), Math.round(this.imag));
    }

    // ---- Comparazione (basata sul modulo) ----
    greaterThan(other: Complex): boolean { return this.magnitude > other.magnitude; }
    lessThan(other: Complex): boolean    { return this.magnitude < other.magnitude; }
    equals(other: Complex): boolean      { return this.real === other.real && this.imag === other.imag; }
    isNearZero(tol: number): boolean     { return this.magnitude < tol; }

    // ---- Coniugato ----
    conjugate(): Complex { return new Complex(this.real, -this.imag); }

    toString(): string {
        if (this.imag === 0) return `${this.real}`;
        const sign = this.imag >= 0 ? "+" : "-";
        return `${this.real} ${sign} ${Math.abs(this.imag)}i`;
    }
}
