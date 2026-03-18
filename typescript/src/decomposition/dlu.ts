import { Matrix } from "../core";

/**
 * Scompone la matrice A in forma additiva: A = D + L + U
 * D: Diagonale
 * L: Triangolare inferiore stretta
 * U: Triangolare superiore stretta
 */
export function decomposeDLU(A: Matrix): { D: Matrix; L: Matrix; U: Matrix } {
    if (A.rows !== A.cols) throw new Error("La matrice deve essere quadrata.");

    const D = new Matrix(A.rows, A.cols);
    const L = new Matrix(A.rows, A.cols);
    const U = new Matrix(A.rows, A.cols);

    for (let i = 1; i <= A.rows; i++) {
        for (let j = 1; j <= A.cols; j++) {
            // Adattamento dell'indice per la lettura in memoria
            const val = A.get(i - 1, j - 1); 
            
            if (i === j) {
                D.set(i - 1, j - 1, val);
            } else if (i > j) {
                L.set(i - 1, j - 1, val);
            } else {
                U.set(i - 1, j - 1, val);
            }
        }
    }
    return { D, L, U };
}