import { Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Neumann
 * * Proprietà:
 * - Corrisponde alla discretizzazione del Laplaciano con condizioni al contorno di Neumann.
 * - È una matrice singolare (il vettore costante è nel nucleo).
 * * Funzionamento:
 * Utilizza una griglia quadrata k x k. Per i nodi ai bordi, applica una riflessione
 * che raddoppia il peso del vicino interno (coefficiente -2 invece di -1).
 */
export function neumann(n: number): Matrix {
    const k = Math.sqrt(n);
    
    if (!Number.isInteger(k)) {
        throw new Error("Il parametro n deve essere un quadrato perfetto (es. 4, 9, 16).");
    }

    const matrix: Matrix = zeros(n, n); 

    for (let p = 1; p <= n; p++) {
        // Calcoliamo le coordinate (i, j) nella griglia k x k (1-based)
        const i = Math.ceil(p / k);
        const j = ((p - 1) % k) + 1;

        // 1. Diagonale principale: sempre 4
        matrix.set(p - 1, p - 1, 4);

        // 2. Vicini Orizzontali (Sinistra/Destra)
        if (k > 1) {
            if (j === 1) {
                // Bordo sinistro: riflette sul vicino a destra (p+1)
                matrix.set(p - 1, (p + 1) - 1, matrix.get(p - 1, (p + 1) - 1) - 2);
            } else if (j === k) {
                // Bordo destro: riflette sul vicino a sinistra (p-1)
                matrix.set(p - 1, (p - 1) - 1, matrix.get(p - 1, (p - 1) - 1) - 2);
            } else {
                // Nodo interno orizzontalmente: entrambi i vicini pesano -1
                matrix.set(p - 1, (p - 1) - 1, matrix.get(p - 1, (p - 1) - 1) - 1);
                matrix.set(p - 1, (p + 1) - 1, matrix.get(p - 1, (p + 1) - 1) - 1);
            }
        }

        // 3. Vicini Verticali (Sopra/Sotto)
        if (k > 1) {
            if (i === 1) {
                // Bordo superiore: riflette sul vicino sotto (p+k)
                matrix.set(p - 1, (p + k) - 1, matrix.get(p - 1, (p + k) - 1) - 2);
            } else if (i === k) {
                // Bordo inferiore: riflette sul vicino sopra (p-k)
                matrix.set(p - 1, (p - k) - 1, matrix.get(p - 1, (p - k) - 1) - 2);
            } else {
                // Nodo interno verticalmente: entrambi i vicini pesano -1
                matrix.set(p - 1, (p - k) - 1, matrix.get(p - 1, (p - k) - 1) - 1);
                matrix.set(p - 1, (p + k) - 1, matrix.get(p - 1, (p + k) - 1) - 1);
            }
        }
    }

    return matrix;
}