// type/interface.ts
export interface INumeric<T> {
    // ---- kind tag: permette fast-path detection senza instanceof ----
    readonly kind: string;

    // ---- Aritmetica ----
    add(other: T): T;
    subtract(other: T): T;
    multiply(other: T): T;
    divide(other: T): T;
    negate(): T;

    // ---- Funzioni matematiche ----
    abs(): T;
    sqrt(): T;
    round(): T;

    /**
     * Coniugato complesso.
     * Per tipi reali (Float64M, Rational) restituisce `this` invariato.
     * Per Complex restituisce il coniugato a-bi.
     * Usato da QR e dalla trasposta coniugata (adjoint).
     */
    conjugate(): T;

    // ---- Comparazione ----
    greaterThan(other: T): boolean;
    lessThan(other: T): boolean;
    equals(other: T): boolean;
    isNearZero(tol: number): boolean;

    // ---- Conversioni ----
    toNumber(): number;
    fromNumber(n: number): T;

    toString(): string;
}
