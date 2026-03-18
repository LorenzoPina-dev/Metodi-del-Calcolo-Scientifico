export interface INumeric {
  add(other: INumeric): INumeric;
  subtract(other: INumeric): INumeric;
  multiply(other: INumeric): INumeric;
  divide(other: INumeric): INumeric;
  greaterThan(other: INumeric): boolean;
  lessThan(other: INumeric): boolean;
  equals(other: INumeric): boolean;
  toString(): string;
}