import { INumeric } from ".";

export class Rational implements INumeric{
  public readonly num: bigint;
  public readonly den: bigint;

  constructor(num: bigint | number, den: bigint | number = 1n) {
    let n = BigInt(num);
    let d = BigInt(den);

    if (d === 0n) throw new Error("Il denominatore non può essere zero.");

    // Portiamo il segno sempre al numeratore
    if (d < 0n) {
      n = -n;
      d = -d;
    }

    // Semplifichiamo la frazione (MCD)
    const common = Rational.gcd(n < 0n ? -n : n, d);
    this.num = n / common;
    this.den = d / common;
  }

  // Algoritmo di Euclide per il Massimo Comune Divisore
  private static gcd(a: bigint, b: bigint): bigint {
    while (b > 0n) {
      a %= b;
      [a, b] = [b, a];
    }
    return a;
  }

  // Operazioni matematiche esatte
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
    return new Rational(this.num * other.den, this.den * other.num);
  }

  // Comparazione
  equals(other: Rational): boolean {
    return this.num === other.num && this.den === other.den;
  }

  greaterThan(other: Rational): boolean {
    return this.num * other.den > other.num * this.den;
  }

  lessThan(other: Rational): boolean {
    return this.num * other.den < other.num * this.den;
  }
  // Output testuale
  toString(): string {
    return this.den === 1n ? `${this.num}` : `${this.num}/${this.den}`;
  }

  // Conversione in decimale (qui l'arrotondamento è inevitabile, ma lo fai solo alla fine)
  toDecimal(precision: number = 10): string {
    const factor = BigInt(10 ** precision);
    const value = (this.num * factor) / this.den;
    const s = value.toString().padStart(precision + 1, '0');
    const dotIndex = s.length - precision;
    return s.slice(0, dotIndex) + "." + s.slice(dotIndex);
  }
}