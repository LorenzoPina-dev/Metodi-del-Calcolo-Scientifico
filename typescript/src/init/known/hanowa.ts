import { Matrix } from "../../core";
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

    for (let i = 1; i <= m; i++) {
        // Blocchi diagonali (d * I)
        A.set(i - 1, i - 1, d);
        A.set(i + m - 1, i + m - 1, d);

        // Blocchi off-diagonal
        A.set(i - 1, i + m - 1, -i); // -diag(1:m)
        A.set(i + m - 1, i - 1, i);  // diag(1:m)
    }

    return A;
}

/*

export function hanowa(n: number, d = -1): Matrix {
    if (n % 2 !== 0) {
        throw new Error("hanowa: n must be even");
    }

    const m = n / 2;
    const A = zeros(n, n);

    // Creiamo le matrici diagonali
    const D = identity(m).multiply(d); // d*eye(m)
    const diag1toM = zeros(m, m);
    for (let i = 0; i < m; i++) {
        diag1toM.set(i, i, i + 1); // diag(1:m)
    }

    // Costruzione dei 4 blocchi
    // Blocchi superiori
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
            A.set(i, j, D.get(i, j));           // D
            A.set(i, j + m, -diag1toM.get(i, j)); // -diag(1:m)
        }
    }

    // Blocchi inferiori
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
            A.set(i + m, j, diag1toM.get(i, j)); // diag(1:m)
            A.set(i + m, j + m, D.get(i, j));    // D
        }
    }

    return A;
}

*/