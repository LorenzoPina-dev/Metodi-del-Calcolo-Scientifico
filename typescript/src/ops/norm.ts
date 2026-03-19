import { Matrix } from "..";

/**
 * Calcola la norma di una matrice o di un vettore.
 * @param type 1, 2, 'inf' (Infinito), 'fro' (Frobenius)
 */
export function norm(this: Matrix, type: 1 | 2 | 'inf' | 'fro' = 2): number {
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

function normVector(A: Matrix, type: any): number {
    const data = A.data;
    const len = data.length;
    let res = 0;

    switch (type) {
        case 1: // Somma dei valori assoluti
            for (let i = 0; i <= len - 4; i += 4) {
                res += Math.abs(data[i]) + Math.abs(data[i+1]) + Math.abs(data[i+2]) + Math.abs(data[i+3]);
            }
            for (let i = len % 4; i > 0; i--) res += Math.abs(data[len - i]);
            return res;

        case 'inf': // Massimo valore assoluto
            let max = 0;
            for (let i = 0; i < len; i++) {
                const abs = Math.abs(data[i]);
                if (abs > max) max = abs;
            }
            return max;

        case 2:
        case 'fro': // Per i vettori, la norma 2 e Frobenius coincidono
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

function normMatrix(A: Matrix, type: any): number {
    switch (type) {
        case 1:
            return norm1Matrix(A); // Max somma assoluta delle colonne
        case 'inf':
            return normInfMatrix(A); // Max somma assoluta delle righe
        case 'fro':
            return normFrobenius(A); // Radice della somma dei quadrati di tutti gli elementi
        case 2:
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
    const data = A.data;
    const colSums = new Float64Array(cols);

    for (let i = 0; i < rows; i++) {
        const offset = i * cols;
        let j = 0;
        for (; j <= cols - 4; j += 4) {
            colSums[j] += Math.abs(data[offset + j]);
            colSums[j+1] += Math.abs(data[offset + j+1]);
            colSums[j+2] += Math.abs(data[offset + j+2]);
            colSums[j+3] += Math.abs(data[offset + j+3]);
        }
        for (; j < cols; j++) colSums[j] += Math.abs(data[offset + j]);
    }

    return Math.max(...colSums);
}

/**
 * Norma Infinito: Massimo della somma assoluta delle righe.
 */
function normInfMatrix(A: Matrix): number {
    const { rows, cols } = A;
    const data = A.data;
    let maxRowSum = 0;

    for (let i = 0; i < rows; i++) {
        const offset = i * cols;
        let rowSum = 0;
        let j = 0;
        for (; j <= cols - 4; j += 4) {
            rowSum += Math.abs(data[offset + j]) + Math.abs(data[offset + j+1]) + 
                      Math.abs(data[offset + j+2]) + Math.abs(data[offset + j+3]);
        }
        for (; j < cols; j++) rowSum += Math.abs(data[offset + j]);
        
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