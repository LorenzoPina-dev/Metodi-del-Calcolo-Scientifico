import { Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Pascal
 * * Proprietà:
 * - Simmetrica e Definita Positiva.
 * - Composta dai coefficienti binomiali del triangolo di Pascal.
 * - Il determinante è sempre 1.
 * - Molto mal condizionata per n grandi.
 * * Funzionamento:
 * Ogni elemento (i,j) è il coefficiente binomiale (i+j-2) su (i-1).
 */
export function pascal(n: number): Matrix {
    if(n > 30) console.warn("[PRECISION_WARNING] n is large, may cause overflow");
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A.set(i, j, nchoosek(i + j, i));
        }
    }
    return A;
}

function nchoosek(n: number, k: number): number {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    if (k > n - k) k = n - k;
    let result = 1;
    for (let i = 1; i <= k; i++) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}