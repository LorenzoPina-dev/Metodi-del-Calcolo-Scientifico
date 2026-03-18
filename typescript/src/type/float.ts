import { INumeric } from ".";

export class Float64M implements INumeric{
  constructor(public readonly value: number) {}

  // 1. Addizione
  add(other: Float64M): Float64M {
    return new Float64M(this.value + other.value);
  }

  // 2. Sottrazione
  subtract(other: Float64M): Float64M {
    return new Float64M(this.value - other.value);
  }

  // 3. Moltiplicazione
  multiply(other: Float64M): Float64M {
    return new Float64M(this.value * other.value);
  }

  // 4. Divisione
  divide(other: Float64M): Float64M {
    if (other.value === 0) throw new Error("Divisione per zero!");
    return new Float64M(this.value / other.value);
  }

  // 5. Comparazione
  greaterThan(other: Float64M): boolean {
    return this.value > other.value;
  }

  lessThan(other: Float64M): boolean {
    return this.value < other.value;
  }

  // Uguaglianza con tolleranza (Epsilon)
  // Nota: Nei float64 è meglio non usare === a causa dei micro-errori di precisione
  equals(other: Float64M): boolean {
    return Math.abs(this.value - other.value) < Number.EPSILON;
  }

  toString(): string {
    return this.value.toString();
  }
}
