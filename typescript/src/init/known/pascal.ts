// init/known/pascal.ts
import { Float64M, Matrix, Rational } from "../..";
import { zeros, zerosLike } from "../init";

// ============================================================
// SOGLIA
// ============================================================
// C(50, 25) ≈ 1.26 × 10¹⁴ — ancora dentro float64 safe integers
// C(60, 30) ≈ 1.18 × 10¹⁷ — supera Number.MAX_SAFE_INTEGER (9 × 10¹⁵)
// Usiamo 26 come soglia conservativa: oltre, ogni elemento potrebbe
// perdere precisione per la catena di moltiplicazioni/divisioni.
const FLOAT_THRESHOLD = 25;

// ============================================================
// OVERLOAD — type safety al call site
// ============================================================

/**
 * Matrice di Pascal: A(i,j) = C(i+j, i). SPD, det = 1.
 *
 * Overload 1 — n ≤ 25: coefficienti float64, prestazioni native.
 * Overload 2 — n > 25 o `mode = 'exact'`: coefficienti Rational esatti
 *              via BigInt. Zero perdita di precisione.
 *
 * Selezione automatica:
 *   pascal(20)          → Matrix<Float64M>   (veloce)
 *   pascal(30)          → Matrix<Rational>   (esatto, auto)
 *   pascal(10, 'exact') → Matrix<Rational>   (esatto, forzato)
 *   pascal(30, 'float') → Matrix<Float64M>   (float, forzato — attenzione overflow)
 */
export function pascal(n: number, mode?: 'float'): Matrix<Float64M>;
export function pascal(n: number, mode: 'exact'): Matrix<Rational>;
export function pascal(
    n: number,
    mode?: 'float' | 'exact'
): Matrix<Float64M> | Matrix<Rational> {

    const useRational = mode === 'exact' || (mode !== 'float' && n > FLOAT_THRESHOLD);

    return useRational
        ? _pascalRational(n)
        : _pascalFloat(n);
}

// ============================================================
// IMPLEMENTAZIONI PRIVATE
// ============================================================

/** Float64M — coefficienti calcolati in floating point. */
function _pascalFloat(n: number): Matrix<Float64M> {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            A.data[i * n + j] = new Float64M(_nchoosekFloat(i + j, i));
    return A;
}

/** Rational — coefficienti esatti via BigInt, zero perdita di precisione. */
function _pascalRational(n: number): Matrix<Rational> {
    const A = zerosLike(n, n, Rational.zero, Rational.one);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            A.data[i * n + j] = new Rational(_nchoosekBigInt(i + j, i), 1n, true);
    return A;
}

// ============================================================
// CALCOLO DEI COEFFICIENTI BINOMIALI
// ============================================================

/**
 * C(n, k) in floating point.
 * Stabile per k piccolo grazie alla simmetria C(n,k) = C(n, n-k).
 */
function _nchoosekFloat(n: number, k: number): number {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    if (k > n - k) k = n - k;
    let r = 1;
    for (let i = 0; i < k; i++) {
        r = r * (n - i) / (i + 1);
    }
    return r;
}

/**
 * C(n, k) esatto in BigInt.
 *
 * Strategia: usa la ricorrenza
 *   C(n, 0) = 1
 *   C(n, i+1) = C(n, i) * (n - i) / (i + 1)
 *
 * Poiché ogni valore intermedio C(n, i) è un intero, la divisione
 * (i+1) è sempre esatta → zero arrotondamento, nessun overflow BigInt.
 */
function _nchoosekBigInt(n: number, k: number): bigint {
    if (k < 0 || k > n) return 0n;
    if (k === 0 || k === n) return 1n;
    if (k > n - k) k = n - k;   // simmetria: minimizza le iterazioni
    let result = 1n;
    for (let i = 0; i < k; i++) {
        // result = C(n, i+1) = C(n, i) * (n - i) / (i + 1)
        // La divisione è sempre esatta perché result è intero.
        result = result * BigInt(n - i) / BigInt(i + 1);
    }
    return result;
}
