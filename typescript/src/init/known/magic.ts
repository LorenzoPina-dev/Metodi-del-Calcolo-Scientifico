import { Matrix } from "../../core";
import { zeros } from "../init";

/**
 * Quadrato Magico
 * * Proprietà:
 * - La somma di ogni riga, colonna e delle due diagonali principali è la stessa.
 * - La costante magica è n*(n^2 + 1)/2.
 * - Per n dispari, n pari (non divisibile per 4) e n doppiamente pari (divisibile per 4)
 * vengono usati algoritmi differenti.
 * * Funzionamento:
 * - Dispari: Metodo Siamese.
 * - Doppiamente Pari: Metodo di inversione dei blocchi 4x4.
 * - Singolarmente Pari: Metodo dei quattro quadranti (Strachey).
 */
export function magic(n: number): Matrix {
    if (n < 3) throw new Error("n must be >= 3");

    if (n % 2 === 1) return magicOdd(n);
    if (n % 4 === 0) return magicDoublyEven(n);

    return magicSinglyEven(n);
}
function magicOdd(n: number): Matrix {
    const M = zeros(n, n);

    let i = 0;
    let j = Math.floor(n / 2);

    for (let k = 1; k <= n * n; k++) {
        M.set(i, j, k);

        const ni = (i - 1 + n) % n;
        const nj = (j + 1) % n;

        if (M.get(ni, nj) !== 0) {
            i = (i + 1) % n;
        } else {
            i = ni;
            j = nj;
        }
    }

    return M;
}
function magicDoublyEven(n: number): Matrix {
    const M = zeros(n, n);

    let num = 1;
    let max = n * n;

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (
                (i % 4 === j % 4) ||
                ((i % 4 + j % 4) === 3)
            ) {
                M.set(i, j, max - num + 1);
            } else {
                M.set(i, j, num);
            }
            num++;
        }
    }

    return M;
}
function magicSinglyEven(n: number): Matrix {
    const p = n / 2;
    const k = (n - 2) / 4;

    const A = magicOdd(p);

    const M = zeros(n, n);

    // blocchi
    for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
            const a = A.get(i, j);

            M.set(i, j, a);
            M.set(i + p, j, a + 3 * p * p);
            M.set(i, j + p, a + 2 * p * p);
            M.set(i + p, j + p, a + p * p);
        }
    }

    // swap colonne
    for (let i = 0; i < p; i++) {
        for (let j = 0; j < k; j++) {
            swap(M, i, j, i + p, j);
        }

        for (let j = n - k + 1; j < n; j++) {
            swap(M, i, j, i + p, j);
        }
    }

    // swap centrale
    swap(M, k, 0, k + p, 0);
    swap(M, k, k, k + p, k);

    return M;
}
function swap(M: Matrix, i1: number, j1: number, i2: number, j2: number) {
    const tmp = M.get(i1, j1);
    M.set(i1, j1, M.get(i2, j2));
    M.set(i2, j2, tmp);
}
