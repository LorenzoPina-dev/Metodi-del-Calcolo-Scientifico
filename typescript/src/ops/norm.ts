import { Matrix } from "..";
type NormType = "1" | "inf" | "fro" | "2";
/**
 * Calcola la norma di una matrice o di un vettore.
 * @param type 1, 2, 'inf' (Infinito), 'fro' (Frobenius)
 */
export function norm(this: Matrix, type: NormType = "2"): number {
    const isVec = this.rows === 1 || this.cols === 1;

    if (isVec) {
        return normVector(this, type);
    } else {
        return normMatrix(this, type);
    }
}

// ============================================================================
// NORME VETTORIALI
// ============================================================================

function normVector(A: Matrix, type: NormType): number {
    const data = A.data;
    const len = data.length;
    let res = 0;

    switch (type.toUpperCase()) {
        case "1": // Somma dei valori assoluti
            for (let i = 0; i <= len - 4; i += 4) {
                res += Math.abs(data[i]) + Math.abs(data[i+1]) + Math.abs(data[i+2]) + Math.abs(data[i+3]);
            }
            for (let i = len % 4; i > 0; i--) res += Math.abs(data[len - i]);
            return res;

        case 'INF': // Massimo valore assoluto
            let max = 0;
            for (let i = 0; i < len; i++) {
                const abs = Math.abs(data[i]);
                if (abs > max) max = abs;
            }
            return max;

        case "2":
        case 'FRO': // Per i vettori, la norma 2 e Frobenius coincidono
            let sumSq = 0;
            for (let i = 0; i <= len - 4; i += 4) {
                sumSq += data[i]**2 + data[i+1]**2 + data[i+2]**2 + data[i+3]**2;
            }
            for (let i = len % 4; i > 0; i--) sumSq += data[len - i]**2;
            return Math.sqrt(sumSq);

        default:
            throw new Error(`Norm type ${type} not supported for vectors.`);
    }
}

// ============================================================================
// NORME MATRICIALI
// ============================================================================

function normMatrix(A: Matrix, type:NormType): number {
    switch (type.toUpperCase()) {
        case "1":
            return norm1Matrix(A); // Max somma assoluta delle colonne
        case 'INF':
            return normInfMatrix(A); // Max somma assoluta delle righe
        case 'FRO':
            return normFrobenius(A); // Radice della somma dei quadrati di tutti gli elementi
        case "2":
            // La norma 2 matriciale è il valore singolare massimo (richiede SVD)
            // Per ora lanciamo un errore o usiamo un'approssimazione se SVD non è pronto
            if ((A as any).svd) {
                const s = (A as any).svd(true) as Matrix; // Richiede SVD economy
                return s.data[0]; 
            }
            throw new Error("Matrix 2-norm requires SVD decomposition.");
        default:
            throw new Error(`Norm type ${type} not supported for matrices.`);
    }
}

/**
 * Norma 1: Massimo della somma assoluta delle colonne.
 */
function norm1Matrix(A: Matrix): number {
    const { rows, cols } = A;
    let max = 0;

    for (let i = 0; i < cols; i++) {
        let sum = 0;
        let j = 0;

        // loop unrolling
        for (; j <= rows - 4; j += 4) {
            sum += Math.abs(A.get(j, i)) + Math.abs(A.get(j + 1, i)) +
                   Math.abs(A.get(j + 2, i)) + Math.abs(A.get(j + 3, i));
        }
        for (; j < rows; j++) sum += Math.abs(A.get(j,i));
        if (sum > max) max = sum;
    }

    return max;
}

/**
 * Norma Infinito: Massimo della somma assoluta delle righe.
 */
function normInfMatrix(A: Matrix): number {
    const { rows, cols } = A;
    let maxRowSum = 0;

    for (let i = 0; i < rows; i++) {
        let rowSum = 0;
        let j = 0;
        for (; j <= cols - 4; j += 4) {
            rowSum += Math.abs(A.get(i, j)) + Math.abs(A.get(i, j+1)) + 
                      Math.abs(A.get(i, j+2)) + Math.abs(A.get(i, j+3));
        }
        for (; j < cols; j++) rowSum += Math.abs(A.get(i, j));
        if (rowSum > maxRowSum) maxRowSum = rowSum;
    }
    return maxRowSum;
}

/**
 * Norma di Frobenius: sqrt(sum(diag(A' * A)))
 */
function normFrobenius(A: Matrix): number {
    const data = A.data;
    const len = data.length;
    let sumSq = 0;
    let i = 0;
    for (; i <= len - 4; i += 4) {
        sumSq += data[i]**2 + data[i+1]**2 + data[i+2]**2 + data[i+3]**2;
    }
    for (; i < len; i++) sumSq += data[i]**2;
    return Math.sqrt(sumSq);
}