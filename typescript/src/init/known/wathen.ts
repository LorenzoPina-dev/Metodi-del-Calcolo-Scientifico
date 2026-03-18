import { Matrix } from "../../core";
import { zeros } from "../init";

/**
 * Matrice di Wathen
 * * Proprietà:
 * - Matrice sparsa derivante da un'applicazione di Elementi Finiti.
 * - Definita Positiva, utilizzata per testare precondizionatori.
 * - La dimensione N è funzione di nx e ny.
 * * Funzionamento:
 * Basata sulla sovrapposizione di matrici di elementi 8x8 pesate da un coefficiente casuale rho.
 */
export function wathen(nx: number, ny: number): Matrix {

    const n = 3 * nx * ny + 2 * nx + 2 * ny + 1;
    const A = zeros(n, n);

    const em = [
        [ 6, -6,  2, -8,  3, -8,  2, -6 ],
        [ -6, 32, -6, 20, -8, 16, -8, 20 ],
        [ 2, -6,  6, -6,  2, -8,  3, -8 ],
        [ -8, 20, -6, 32, -6, 20, -8, 16 ],
        [ 3, -8,  2, -6,  6, -6,  2, -8 ],
        [ -8, 16, -8, 20, -6, 32, -6, 20 ],
        [ 2, -8,  3, -8,  2, -6,  6, -6 ],
        [ -6, 20, -8, 16, -8, 20, -6, 32 ]
    ];

    const node = new Array<number>(8);

    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {

            // ⚠️ Traduzione 1:1 (attenzione agli indici)
            node[0] = (3 * (j + 1)) * nx + 2 * (j + 1) + 2 * (i + 1) + 1 - 1;
            node[1] = node[0] - 1;
            node[2] = node[0] - 2;

            node[3] = (3 * (j + 1) - 1) * nx + 2 * (j + 1) + (i + 1) - 1 - 1;
            node[7] = node[3] + 1;

            node[4] = (3 * (j + 1) - 3) * nx + 2 * (j + 1) + 2 * (i + 1) - 3 - 1;
            node[5] = node[4] + 1;
            node[6] = node[4] + 2;

            const rho = 100 * Math.random();

            for (let krow = 0; krow < 8; krow++) {
                for (let kcol = 0; kcol < 8; kcol++) {
                    const r = node[krow];
                    const c = node[kcol];
                    A.set(r, c, A.get(r, c) + rho * em[krow][kcol]);
                }
            }
        }
    }

    return A;
}