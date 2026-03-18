import { Matrix } from "../../core";
import { zeros } from "../init";

/**
 * Matrice Min(i,j)
 * * Descrizione:
 * Matrice simmetrica definita dagli indici minimi: A(i,j) = min(i,j).
 * * Proprietà:
 * - Simmetrica e Definita Positiva (SPD).
 * - La sua inversa è una matrice tridiagonale molto semplice.
 * - Corrisponde alla matrice di covarianza di un processo di Wiener (moto browniano) campionato.
 * * Funzionamento:
 * Riempie la matrice confrontando gli indici di riga e colonna (1-based).
 */
export function minij(n: number): Matrix {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A.set(i, j, Math.min(i + 1, j + 1)); // +1 perché R usa indici 1-based
        }
    }
    return A;
}