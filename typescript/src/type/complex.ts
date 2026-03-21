import { INumeric } from "./interface";


export class Complex implements INumeric<Complex> {
  constructor(public readonly real: number, public readonly imag: number) {}
  static get zero(): Complex { return new Complex(0, 0); }
  static get one(): Complex { return new Complex(1, 0); }


  // Operazioni matematiche esatte per numeri complessi

  // 1. Addizione
  add(other: Complex): Complex {
    return new Complex(this.real + other.real, this.imag + other.imag);
  }

  // 2. Sottrazione
  subtract(other: Complex): Complex {
    return new Complex(this.real - other.real, this.imag - other.imag);
  }

  // 3. Moltiplicazione
  multiply(other: Complex): Complex {
    return new Complex(
      this.real * other.real - this.imag * other.imag,
      this.real * other.imag + this.imag * other.real
    );
  }

  // 4. Divisione
  divide(other: Complex): Complex {
    const denominator = other.real ** 2 + other.imag ** 2;
    if (denominator === 0) throw new Error("Divisione per zero!");
    
    const realPart = (this.real * other.real + this.imag * other.imag) / denominator;
    const imagPart = (this.imag * other.real - this.real * other.imag) / denominator;
    return new Complex(realPart, imagPart);
  }

  // 5. Modulo (Magnitudine)
  get magnitude(): number {
    return Math.sqrt(this.real ** 2 + this.imag ** 2);
  }
  abs(): Complex {
    return new Complex(this.magnitude, 0);
  }

  // 6. Comparazione (basata sul modulo)
  greaterThan(other: Complex): boolean {
    return this.magnitude > other.magnitude;
  }

  lessThan(other: Complex): boolean {
    return this.magnitude < other.magnitude;
  }

  equals(other: Complex): boolean {
    return this.real === other.real && this.imag === other.imag;
  }

  round(): Complex {
    return new Complex(Math.round(this.real), Math.round(this.imag));
  }
  sqrt(): Complex {
    const r = this.magnitude;
    const theta = Math.atan2(this.imag, this.real);
    return new Complex(Math.sqrt(r) * Math.cos(theta / 2), Math.sqrt(r) * Math.sin(theta / 2));
  } 


  // Utility per stampare il numero
  toString(): string {
    const sign = this.imag >= 0 ? "+" : "-";
    return `${this.real} ${sign} ${Math.abs(this.imag)}i`;
  }
}