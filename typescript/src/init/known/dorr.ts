import { Matrix } from "../../core";
import { zeros } from "../init";

/**
 * Matrice di Dorr
 * * Proprietà:
 * - Matrice tridiagonale, diagonalmente dominante.
 * - Per valori piccoli di theta, il sistema lineare diventa molto mal condizionato.
 * * Funzionamento:
 * Discretizza un problema ai valori al contorno singolarmente perturbato.
 * Divide la costruzione in due blocchi (1:m e m+1:n).
 */
export function dorr(n: number, theta = 0.01): Matrix {
    const h = 1 / (n + 1);
    const m = Math.floor((n + 1) / 2);
    const term = theta / (h * h);

    const c = new Array(n).fill(0); // subdiagonal
    const e = new Array(n).fill(0); // superdiagonal
    const d = new Array(n).fill(0); // main diagonal

    // Primo blocco i = 1:m
    for (let i = 0; i < m; i++) {
        c[i] = -term;
        e[i] = c[i] - (0.5 - (i + 1) * h) / h;
        d[i] = -(c[i] + e[i]);
    }

    // Secondo blocco i = m+1:n
    for (let i = m; i < n; i++) {
        e[i] = -term;
        c[i] = e[i] + (0.5 - (i + 1) * h) / h;
        d[i] = -(c[i] + e[i]);
    }

    // Ridimensionamento dei vettori per la tridiagonale
    const sub = c.slice(1, n);      // subdiagonale n-1
    const sup = e.slice(0, n - 1);  // superdiagonale n-1

    // Costruzione matrice tridiagonale
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        A.set(i, i, d[i]);
        if (i < n - 1) {
            A.set(i, i + 1, sup[i]);
            A.set(i + 1, i, sub[i]);
        }
    }

    return A;
}