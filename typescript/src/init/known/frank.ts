import { Matrix } from "../../core";
import { triu } from "../../decomposition/triu";
import { zeros } from "../init";
import { minij } from "./minij";

/**
 * Matrice di Frank
 * * Descrizione:
 * Una matrice di Hessenberg superiore con determinante 1.
 * * Proprietà:
 * - Gli autovalori minori sono molto sensibili alle perturbazioni.
 * - Gli autovalori sono reali e positivi, e si presentano in coppie reciproche se n è pari.
 * * Funzionamento:
 * Costruisce una matrice basata su min(i,j), ne estrae la parte triangolare superiore
 * estesa alla prima sottodiagonale e opzionalmente la riflette.
 */
export function frank(n: number, k = 0): Matrix {
    let F = minij(n);
    F = triu(F, -1);

    if (k === 0) {
        const p = Array.from({ length: n }, (_, i) => n - i - 1); // R indice inverso 1:n
        const G = zeros(n, n);
        // Riflettiamo sulla anti-diagonale
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                G.set(i, j, F.get(p[i], p[j]));
            }
        }
        // Trasponiamo
        const H = zeros(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                H.set(i, j, G.get(j, i));
            }
        }
        F = H;
    }

    return F;
}

/*
// Funzione Frank
export function frank(n: number, k = 0): Matrix {
    let F = minij(n);
    F = triu(F, -1);

    if (k === 0) {
        const p = Array.from({ length: n }, (_, i) => n - i - 1); // R indice inverso 1:n
        const G = zeros(n, n);
        // Riflettiamo sulla anti-diagonale
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                G.set(i, j, F.get(p[i], p[j]));
            }
        }
        // Trasponiamo
        const H = zeros(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                H.set(i, j, G.get(j, i));
            }
        }
        F = H;
    }

    return F;
}
*/