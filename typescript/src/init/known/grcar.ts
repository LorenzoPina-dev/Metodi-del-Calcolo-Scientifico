import { Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Grcar
 * * Proprietà:
 * - Matrice di Toeplitz non simmetrica.
 * - Possiede autovalori molto sensibili alle perturbazioni (punti nel piano complesso).
 * * Funzionamento:
 * Imposta la diagonale principale e k sovradiagonali a 1.
 * Imposta la sottodiagonale a -1.
 */
export function grcar(n: number, k: number = 3): Matrix {
    const G = zeros(n, n);
    for (let i = 0; i <= n; i++) {
        G.set(i, i, 1);
        for (let j = 1; j <= k; j++) {
            if (i + j < n) G.set(i, i + j, 1);
        }
        if (i > 0) G.set(i, i - 1, -1);
    }
    return G;
}