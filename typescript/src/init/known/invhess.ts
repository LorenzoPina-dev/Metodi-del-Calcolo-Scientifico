import { Matrix } from "../../core";
import { zeros } from "../init";

/**
 * Matrice Inverse Hessenberg
 * * Descrizione:
 * Una matrice densa la cui struttura riflette l'inversa di una matrice di Hessenberg superiore.
 * * Proprietà:
 * - La struttura dei segni cambia tra la parte inferiore (inclusa diagonale) e superiore.
 * - Elemento (i,j) con j < i: valore (j).
 * - Elemento (i,j) con i = j: valore (i).
 * - Elemento (i,j) con j > i: valore (-i).
 * * Funzionamento:
 * Utilizza una doppia iterazione basata su indici 1-based per determinare il valore
 * e il segno di ogni cella.
 */
export function invhess(n: number): Matrix {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (j < i) {
                A.set(i, j, j+1);      // sotto diagonale
            } else if (i === j) {
                A.set(i, j, i + 1);  // diagonale principale
            } else {
                A.set(i, j, -(i + 1)); // sopra diagonale
            }
        }
    }

    return A;
}