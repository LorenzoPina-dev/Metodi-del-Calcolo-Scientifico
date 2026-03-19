import { Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Lehmer
 * * Proprietà:
 * - Simmetrica e Definita Positiva.
 * - L'inversa è una matrice tridiagonale.
 * - Gli elementi sono A(i,j) = min(i,j) / max(i,j).
 * * Funzionamento:
 * Calcola il rapporto tra il valore minimo e massimo degli indici correnti (1-based).
 */
export function lehmer(n: number): Matrix {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A.set(i, j,  Math.min(i + 1, j + 1) / Math.max(i +1, j + 1));
        }
    }
    return A;
}