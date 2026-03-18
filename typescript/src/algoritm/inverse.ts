import { Matrix } from "../core";
import { identity } from "../init";
import { solve } from "../solver";

export function inverseDiagonal(A: Matrix): Matrix {
        const out = new Matrix(A.rows, A.cols);
        for (let i = 1; i <= A.rows; i++) {
            const val = A.get(i - 1, i - 1);
            if (Matrix.isZero(val)) throw new Error("Matrice diagonale singolare.");
            out.set(i - 1, i - 1, 1 / val);
        }
        return out;
    }

    
    export function inverse(A: Matrix): Matrix {
        return solve(A, identity(A.rows));
    }
   export function inverseOrthogonal(A: Matrix): Matrix {
        // In una matrice ortogonale, l'inversa è la trasposta
        return A.transpose();
    }

   export function pseudoInverse(A: Matrix): Matrix {
        const At = A.transpose();
        // (A^T * A)^-1 * A^T
        return inverse(At.multiply(A)).multiply(At);
    }

   export function inverseTriangular(A: Matrix, type: "upper" | "lower"): Matrix {
        const n = A.rows;
        const out = new Matrix(n, n);
        const isUpper = type === "upper";

        for (let j = 1; j <= n; j++) {
            // Risolviamo A * x = e_j (colonna j dell'identità)
            if (isUpper) {
                for (let i = n; i >= 1; i--) {
                    let sum = 0;
                    for (let k = i + 1; k <= n; k++) {
                        sum += A.get(i - 1, k - 1) * out.get(k - 1, j - 1);
                    }
                    const diag = A.get(i - 1, i - 1);
                    const b = (i === j ? 1 : 0);
                    out.set(i - 1, j - 1, (b - sum) / diag);
                }
            } else {
                for (let i = 1; i <= n; i++) {
                    let sum = 0;
                    for (let k = 1; k <= i - 1; k++) {
                        sum += A.get(i - 1, k - 1) * out.get(k - 1, j - 1);
                    }
                    const diag = A.get(i - 1, i - 1);
                    const b = (i === j ? 1 : 0);
                    out.set(i - 1, j - 1, (b - sum) / diag);
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
        if (n !== m) {
            console.log("SmartInverse: Matrice rettangolare rilevata. Uso Moore-Penrose.");
            return pseudoInverse(A);
        }

        // 2. Controllo Diagonale / Scalare
        let isDiag = true;
        const firstDiag = A.get(0, 0);

        for (let i = 1; i <= n; i++) {
            for (let j = 1; j <= n; j++) {
                const val = A.get(i - 1, j - 1);
                if (i !== j && !Matrix.isZero(val)) {
                    isDiag = false;
                    break;
                }
            }
            if (!isDiag) break;
        }

        if (isDiag) {
            console.log("SmartInverse: Matrice diagonale rilevata.");
            return inverseDiagonal(A);
        }

        // 3. Controllo Triangolare
        let isUpper = true;
        let isLower = true;
        for (let i = 1; i <= n; i++) {
            for (let j = 1; j <= n; j++) {
                const val = A.get(i - 1, j - 1);
                if (i > j && !Matrix.isZero(val)) isUpper = false;
                if (i < j && !Matrix.isZero(val)) isLower = false;
            }
        }

        if (isUpper) {
            console.log("SmartInverse: Matrice triangolare superiore rilevata.");
            return inverseTriangular(A,"upper");
        }
        if (isLower) {
            console.log("SmartInverse: Matrice triangolare inferiore rilevata.");
            return inverseTriangular(A,"lower");
        }

        // 4. Controllo Ortogonale (A^T * A = I)
        // Nota: Questo controllo costa O(n^3), ha senso solo se prevedi molte matrici di rotazione.
        // Se la matrice è grande, conviene saltarlo o farlo solo su richiesta.
        if (A.isActuallyOrthogonal()) {
            console.log("SmartInverse: Matrice ortogonale rilevata.");
            return A.transpose();
        }

        // 5. Fallback: Metodo Diretto LUP
        console.log("SmartInverse: Nessun pattern specifico. Uso decomposizione LUP.");
        return inverse(A);
    }