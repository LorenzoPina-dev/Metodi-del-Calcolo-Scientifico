export interface INumeric<T> {
  abs(): T;
  sqrt(): T;
  round(): T;
  add(other: T): T;
  subtract(other: T): T;
  multiply(other: T): T;
  divide(other: T): T;
  greaterThan(other: T): boolean;
  lessThan(other: T): boolean;
  equals(other: T): boolean;
  toString(): string;
}