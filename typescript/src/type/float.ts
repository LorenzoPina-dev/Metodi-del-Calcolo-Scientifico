import { INumeric } from ".";

export class Float64M implements INumeric<Float64M> {
  constructor(public readonly value: number) {}
  static get zero(): Float64M { return new Float64M(0); }
  static get one(): Float64M { return new Float64M(1); }

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
  abs(): Float64M {
    return new Float64M(Math.abs(this.value));
  }
  sqrt(): Float64M {
    return new Float64M(Math.sqrt(this.value));
  }
  round(): Float64M {
    return new Float64M(Math.round(this.value));
  }

  toString(): string {
    return this.value.toString();
  }
}
