import { Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice Tridiagonale
 * * Descrizione:
 * Una matrice che presenta elementi diversi da zero solo sulla diagonale principale, 
 * sulla prima sottodiagonale e sulla prima sovradiagonale.
 * * Proprietà:
 * - Altamente sparsa (contiene al massimo 3n-2 elementi non nulli).
 * - Facile da invertire tramite l'algoritmo di Thomas (TDMA).
 * - Spesso derivata dalla discretizzazione di derivate seconde (ODE/PDE).
 * * Funzionamento:
 * Accetta tre array: a (sotto, n-1), b (principale, n), c (sopra, n-1).
 */
export function tridiag(a: number[], b: number[], c: number[]): Matrix {
    const n = b.length;
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        A.set(i, i, b[i]);
        if (i > 0) A.set(i, i - 1, a[i - 1]);
        if (i < n - 1) A.set(i, i + 1, c[i]);
    }
    return A;
}