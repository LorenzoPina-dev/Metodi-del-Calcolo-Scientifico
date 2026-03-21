// type/interface.ts

/**
 * Interfaccia generica per un tipo numerico.
 * Ogni implementazione (Float64M, Complex, Rational) deve soddisfare questo contratto.
 * Le operazioni aritmetiche sono immutabili: restituiscono sempre una nuova istanza.
 */
export interface INumeric<T> {
    // ---- Operazioni aritmetiche ----
    add(other: T): T;
    subtract(other: T): T;
    multiply(other: T): T;
    divide(other: T): T;
    negate(): T;

    // ---- Funzioni matematiche ----
    abs(): T;
    sqrt(): T;
    round(): T;

    // ---- Comparazione ----
    greaterThan(other: T): boolean;
    lessThan(other: T): boolean;
    equals(other: T): boolean;

    /**
     * Verifica se il valore è vicino a zero entro una tolleranza.
     * Per tipi esatti (Rational) ignora tol e controlla l'esatto zero.
     */
    isNearZero(tol: number): boolean;

    // ---- Conversioni ----
    /** Converte in number JS (per norme, det float, ecc.). */
    toNumber(): number;

    /**
     * Factory: crea una nuova istanza del medesimo tipo a partire da un number JS.
     * Permette alle funzioni generiche di costruire valori senza conoscere T.
     * Esempio: `zero.fromNumber(3.14)` restituisce un Float64M(3.14) o Rational(3.14) ecc.
     */
    fromNumber(n: number): T;

    // ---- Presentazione ----
    toString(): string;
}
