import { Matrix } from '..';
import { lup } from '../decomposition';

/**
 * Calcola il determinante della matrice in modo ottimizzato.
 * @param A Istanza della matrice
 * @returns number
 */
export function det(A: Matrix): number {
    if (!A.isSquare()) {
        throw new Error("Il determinante è definito solo per matrici quadrate.");
    }

    const n = A.rows;

    // 1. Caso banale 1x1
    if (n === 1) {
        return A.get(1, 1);
    }

    // 2. Caso 2x2 (Formula diretta più veloce)
    if (n === 2) {
        return det2x2(A);
    }
    // 3. Matrici Diagonali o Triangolari (Prodotto della diagonale)
    if (A.isDiagonal() || A.isUpperTriangular() || A.isLowerTriangular()) {
        return detTriangular(A);
    }

    // 4. Caso Generale (Decomposizione LU)
    return detGeneral(A);
}

/**
 * Algoritmo per matrici 2x2: ad - bc
 */
function det2x2(A: Matrix): number {
    // Utilizziamo indicizzazione 1-based come richiesto
    return A.get(1, 1) * A.get(2, 2) - A.get(1, 2) * A.get(2, 1);
}

/**
 * Algoritmo per matrici triangolari o diagonali: Prodotto degli elementi diagonali
 * Complessità: O(n)
 */
function detTriangular(A: Matrix): number {
    let d = 1;
    const n = A.rows;
    for (let i = 0; i < n; i++) {
        const val = A.get(i, i);
        if (val === 0) return 0; // Early exit: se c'è uno zero sulla diagonale, det = 0
        d *= val;
    }
    return d;
}

/**
 * Algoritmo Generale: utilizza la decomposizione LU con pivoting (LUP)
 * det(A) = det(P⁻¹LU) = det(P⁻¹) * det(L) * det(U)
 * det(L) è sempre 1 (diagonale di uni), det(P⁻¹) è (-1)^s dove s è il numero di scambi.
 * Complessità: O(n³)
 */
 function detGeneral(A: Matrix): number {
    // Sfruttiamo la funzione LU già presente nella tua cartella decomposition
    // Assumiamo che ritorni { L, U, P, swaps }
    const { U, swaps } = lup(A); 
    
    let d = Math.pow(-1, swaps);
    const n = U.rows;

    for (let i = 0; i < n; i++) {
        const val = U.get(i, i);
        if (Math.abs(val) < 1e-15) return 0; // Matrice singolare o quasi singolare
        d *= val;
    }

    return d;
}