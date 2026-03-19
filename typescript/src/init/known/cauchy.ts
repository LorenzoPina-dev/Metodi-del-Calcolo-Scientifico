import { Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Cauchy
 * * Proprietà:
 * - Definita da due vettori x e y.
 * - Elemento C(i,j) = 1 / (x[i] + y[j]).
 * - Molte matrici famose (come la Hilbert) sono casi speciali della Cauchy.
 * * Funzionamento:
 * Riceve due array di coordinate e genera una matrice n x m.
 * Lancia un errore se la somma di una coppia x_i + y_j è zero.
 */
export function cauchy(x: number[], y: number[]): Matrix {
    const n = x.length;
    const m = y.length;
    const C = zeros(n, m);

    for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
            const denom = x[i] + y[j];
            if (Math.abs(denom) < Matrix.EPS)
                throw new Error("Division by zero in Cauchy matrix");
            C.set(i, j, 1 / denom);
        }

    return C;
}