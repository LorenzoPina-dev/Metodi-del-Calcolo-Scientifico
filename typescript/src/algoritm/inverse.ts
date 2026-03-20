import { Matrix } from "..";
import { identity } from "../init";
import { solve } from "../solver";

export function inverseDiagonal(A: Matrix): Matrix {
        const out = new Matrix(A.rows, A.cols);
        for (let i = 0; i < A.rows; i++) {
            const val = A.get(i, i);
            if (Matrix.isZero(val)) throw new Error("Matrice diagonale singolare.");
            out.set(i, i, 1 / val);
        }
        return out;
    }

    
    export function inverse(A: Matrix): Matrix {
        return solve(A, identity(A.rows));
    }
   export function inverseOrthogonal(A: Matrix): Matrix {
        // In una matrice ortogonale, l'inversa è la trasposta
        return A.t();
    }

   export function pseudoInverse(A: Matrix): Matrix {
        const At = A.t();
        // (A^T * A)^-1 * A^T
        return inverse(At.mul(A)).mul(At);
    }

   export function inverseTriangular(A: Matrix, type: "upper" | "lower"): Matrix {
        const n = A.rows;
        const out = new Matrix(n, n);
        const isUpper = type === "upper";

        for (let j = 0; j < n; j++) {
            // Risolviamo A * x = e_j (colonna j dell'identità)
            if (isUpper) {
                for (let i = n - 1; i >= 0; i--) {
                    let sum = 0;
                    for (let k = i + 1; k < n; k++) {
                        sum += A.get(i, k) * out.get(k, j);
                    }
                    const diag = A.get(i, i);
                    const b = (i === j ? 1 : 0);
                    out.set(i, j, (b - sum) / diag);
                }
            } else {
                for (let i = 0; i < n; i++) {
                    let sum = 0;
                    for (let k = 0; k < i; k++) {
                        sum += A.get(i, k) * out.get(k, j);
                    }
                    const diag = A.get(i, i);
                    const b = (i === j ? 1 : 0);
                    out.set(i, j, (b - sum) / diag);
                }
            }
        }
        return out;
    }

    /**
     * Analizza la matrice e sceglie l'algoritmo di inversione più efficiente.
     * Ordine di controllo: Scalare -> Diagonale -> Ortogonale -> Triangolare -> Quadrata -> Rettangolare.
     */
   export function smartInverse(A: Matrix): Matrix {
        const n = A.rows;
        const m = A.cols;

        // 1. Caso Rettangolare (Pseudo-inversa)
        if (!A.isSquare()) {
            console.log("SmartInverse: Matrice rettangolare rilevata. Uso Moore-Penrose.");
            return pseudoInverse(A);
        }

        if (A.isDiagonal()) {
            console.log("SmartInverse: Matrice diagonale rilevata.");
            return inverseDiagonal(A);
        }

        if (A.isUpperTriangular()) {
            console.log("SmartInverse: Matrice triangolare superiore rilevata.");
            return inverseTriangular(A,"upper");
        }
        if (A.isLowerTriangular()) {
            console.log("SmartInverse: Matrice triangolare inferiore rilevata.");
            return inverseTriangular(A,"lower");
        }

        // 4. Controllo Ortogonale (A^T * A = I)
        // Nota: Questo controllo costa O(n^3), ha senso solo se prevedi molte matrici di rotazione.
        // Se la matrice è grande, conviene saltarlo o farlo solo su richiesta.
        if (A.isOrthogonal()) {
            console.log("SmartInverse: Matrice ortogonale rilevata.");
            return A.t();
        }

        // 5. Fallback: Metodo Diretto LUP
        console.log("SmartInverse: Nessun pattern specifico. Uso decomposizione LUP.");
        return inverse(A);
    }