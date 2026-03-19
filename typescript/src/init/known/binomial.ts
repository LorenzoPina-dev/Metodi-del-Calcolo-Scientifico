import { Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice Binomiale
 * * Descrizione:
 * Matrice i cui elementi sono calcolati tramite somme pesate di coefficienti binomiali.
 * Equivalente a gallery('binomial', n) in MATLAB.
 * * Proprietà:
 * - È una matrice involutoria (il suo quadrato è un multiplo dell'identità) se definita correttamente.
 * - Gli elementi possono diventare molto grandi rapidamente, causando overflow per n > 30.
 * * Funzionamento:
 * Calcola ogni cella come $\sum_{k=1}^n (-1)^{k-1} \binom{i-1}{k-1} \binom{n-i}{j-k}$.
 */
export function binomial(n: number): Matrix {
    const matrix: Matrix = zeros(n, n);

    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= n; j++) {
            let sum = 0;
            for (let k = 1; k <= n; k++) {
                const term1 = combinations(i - 1, k - 1);
                const term2 = combinations(n - i, j - k);
                const sign = (k - 1) % 2 === 0 ? 1 : -1;
                
                sum += term1 * term2 * sign;
            }
            matrix.set(i - 1, j - 1, sum);
        }
    }

    return matrix;
}

/**
 * Calcola il coefficiente binomiale (n su k)
 * Implementa la simmetria per ottimizzare i calcoli.
 */
function combinations(n: number, k: number): number {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    if (k > n / 2) k = n - k;
    let res = 1;
    for (let i = 1; i <= k; i++) {
        res = res * (n - i + 1) / i;
    }
    return Math.round(res);
}