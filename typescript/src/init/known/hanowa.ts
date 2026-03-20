import { Matrix } from "../..";
import { zeros, identity } from "../init";

/**
 * Matrice di Hanowa
 * * Descrizione:
 * Una matrice a blocchi 2x2 di dimensione n (con n pari).
 * * Struttura:
 * [ d*I, -diag(1:m) ]
 * [ diag(1:m), d*I  ]
 * * Proprietà:
 * - Gli autovalori sono complessi: d ± i*k per k=1:m.
 * * Funzionamento:
 * Divide n in due blocchi m = n/2 e riempie i quattro quadranti come descritto.
 */
export function hanowa(n: number, d = -1): Matrix {
    if (n % 2 !== 0) throw new Error("hanowa: n must be even");

    const m = n / 2;
    const A = zeros(n, n);

    for (let i = 0; i < m; i++) {
        // Blocchi diagonali (d * I)
        A.set(i, i, d);
        A.set(i + m, i + m, d);

        // Blocchi off-diagonal
        A.set(i, i + m, -(i + 1)); // -diag(1:m)
        A.set(i + m, i, i + 1);  // diag(1:m)
    }

    return A;
}
