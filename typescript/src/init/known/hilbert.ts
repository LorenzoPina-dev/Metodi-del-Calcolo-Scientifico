import { Matrix } from "../../core";
import { zeros } from "../init";

/**
 * Matrice di Hilbert
 * * Proprietà:
 * - Simmetrica e Definita Positiva (SPD).
 * - Estremamente mal condizionata (il numero di condizionamento cresce esponenzialmente con n).
 * - Gli elementi sono definiti come H(i,j) = 1 / (i + j - 1).
 * * Funzionamento:
 * Utilizza una doppia iterazione basata su indici 1-based per calcolare il reciproco
 * della somma degli indici meno uno.
 */
export function hilbert(n: number): Matrix {
    const H = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            H.set(i, j, 1 / (i + j + 1));
    return H;
}