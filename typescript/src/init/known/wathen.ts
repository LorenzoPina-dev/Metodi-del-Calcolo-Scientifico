import { Matrix } from "../../";
import { zeros } from "../init";

/**
 * Matrice di Wathen
 * * Descrizione:
 * Matrice sparsa definita positiva derivante da elementi finiti (elementi isoparametrici a 8 nodi).
 * La dimensione totale n è data da: 3*nx*ny + 2*nx + 2*ny + 1.
 */
export function wathen(nx: number, ny: number): Matrix {
    const n = 3 * nx * ny + 2 * nx + 2 * ny + 1;
    const A = zeros(n, n);

    // Matrice dell'elemento locale (8x8)
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

    // Cicli 0-based per gli elementi della griglia
    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
            
            // --- Mappatura dei nodi (0-based) ---
            // Traduzione semplificata della formula di Wathen per eliminare gli offset
            node[0] = 3 * (j + 1) * nx + 2 * (j + 1) + 2 * i + 2;
            node[1] = node[0] - 1;
            node[2] = node[0] - 2;

            node[3] = (3 * (j + 1) - 1) * nx + 2 * j + i + 1;
            node[7] = node[3] + 1;

            node[4] = (3 * j) * nx + 2 * j + 2 * i;
            node[5] = node[4] + 1;
            node[6] = node[4] + 2;

            // Coefficiente di densità casuale (rho)
            const rho = 100 * Math.random();

            // Assemblaggio della matrice globale
            for (let krow = 0; krow < 8; krow++) {
                for (let kcol = 0; kcol < 8; kcol++) {
                    const r = node[krow];
                    const c = node[kcol];
                    
                    // A[r,c] += rho * em[krow,kcol]
                    const currentVal = A.get(r, c);
                    A.set(r, c, currentVal + rho * em[krow][kcol]);
                }
            }
        }
    }

    return A;
}