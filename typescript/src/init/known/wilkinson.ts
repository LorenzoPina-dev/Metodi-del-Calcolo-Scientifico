import { Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Wilkinson (Wn+)
 * * Descrizione:
 * Matrice tridiagonale simmetrica con coppie di autovalori quasi identici.
 * * Struttura:
 * - Diagonale principale: [m, m-1, ..., 0, ..., m-1, m] dove m = (n-1)/2.
 * - Codiagonali: Tutti 1.
 */
export function wilkinson(n: number): Matrix {
    const A = zeros(n, n);
    const m = (n - 1) / 2;

    for (let i = 0; i < n; i++) {
        // 1. Diagonale principale: valore assoluto della distanza dal centro
        // Se n=7, m=3. La diagonale sarà: |3-0|, |3-1|, |3-2|, |3-3|, |3-4|... -> [3, 2, 1, 0, 1, 2, 3]
        const diagVal = Math.abs(m - i);
        A.set(i, i, diagVal);

        // 2. Codiagonali (Sopra e Sotto)
        if (i < n - 1) {
            A.set(i, i + 1, 1); // Superiore
            A.set(i + 1, i, 1); // Inferiore
        }
    }

    return A;
}