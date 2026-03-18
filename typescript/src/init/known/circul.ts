import { Matrix } from "../../core";
import { zeros } from "../init";

/**
 * Matrice Circolante
 * * Proprietà:
 * - Ogni riga è uno shift ciclico a destra della riga precedente.
 * - Diagonalizzata dalla Trasformata di Fourier Discreta (DFT).
 * * Funzionamento:
 * Prende un vettore v e lo usa come prima riga, traslando gli elementi
 * circolarmente per le righe successive tramite l'operatore modulo.
 */
export function circul(v: number[]): Matrix {
    const n = v.length;
    const C = zeros(n, n);

    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            C.set(i, j, v[(j - i + n) % n]);

    return C;
}