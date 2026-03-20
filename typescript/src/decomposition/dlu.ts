import { Matrix } from "..";

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

    for (let i = 0; i < A.rows; i++) {
        for (let j = 0; j < A.cols; j++) {
            // Adattamento dell'indice per la lettura in memoria
            const val = A.get(i, j); 
            
            if (i === j) {
                D.set(i, j, val);
            } else if (i > j) {
                L.set(i, j, val);
            } else {
                U.set(i, j, val);
            }
        }
    }
    return { D, L, U };
}