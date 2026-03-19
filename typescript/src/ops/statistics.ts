import { Matrix  } from "..";


type CompareFn = (newVal: number, best: number) => boolean;

/**
 * MAX - MATLAB style
 */

export function max(this: Matrix, dim: 1 | 2 = 1): { value: Matrix; index: Int32Array } {
    const comp: CompareFn = (newVal, best) => newVal > best;
    return dim === 1 
        ? computeColumnReduction(this, -Infinity, comp)
        : computeRowReduction(this, -Infinity, comp);
}

/**
 * MIN - MATLAB style
 */
export function min(this: Matrix, dim: 1 | 2 = 1): { value: Matrix; index: Int32Array } {
    const comp: CompareFn = (newVal, best) => newVal < best;
    return dim === 1 
        ? computeColumnReduction(this, Infinity, comp)
        : computeRowReduction(this, Infinity, comp);
}

/**
 * Riduttore per Colonne (dim 1)
 * Risultato: 1 x C
 */
function computeColumnReduction(A: Matrix, initialValue: number, compare: CompareFn): { value: Matrix; index: Int32Array } {
    const R = A.rows, C = A.cols, Adat = A.data;
    const vOut = new Matrix(1, C);
    const iOut = new Int32Array(C);
    
    vOut.data.fill(initialValue);

    for (let i = 0; i < R; i++) {
        const off = i * C;
        for (let j = 0; j < C; j++) {
            const val = Adat[off + j];
            if (compare(val, vOut.data[j])) {
                vOut.data[j] = val;
                iOut[j] = i + 1; // Base 1 come richiesto
            }
        }
    }
    return { value: vOut, index: iOut };
}

/**
 * Riduttore per Righe (dim 2)
 * Risultato: R x 1
 */
function computeRowReduction(A: Matrix, initialValue: number, compare: CompareFn): { value: Matrix; index: Int32Array } {
    const R = A.rows, C = A.cols, Adat = A.data;
    const vOut = new Matrix(R, 1);
    const iOut = new Int32Array(R);

    for (let i = 0; i < R; i++) {
        const off = i * C;
        let bestVal = initialValue;
        let bestIdx = 0;
        
        for (let j = 0; j < C; j++) {
            const val = Adat[off + j];
            if (compare(val, bestVal)) {
                bestVal = val;
                bestIdx = j + 1; // Base 1
            }
        }
        vOut.data[i] = bestVal;
        iOut[i] = bestIdx;
    }
    return { value: vOut, index: iOut };
}


// ============================================================================
// METODI PUBBLICI
// ============================================================================

/**
 * Somma degli elementi lungo la dimensione specificata.
 * dim 1: somma le colonne (risultato 1xN)
 * dim 2: somma le righe (risultato Mx1)
 */
export function sum(this: Matrix, dim: 1 | 2 = 1): Matrix {
    return dim === 1 ? sumColumns(this) : sumRows(this);
}

/**
 * Media degli elementi lungo la dimensione specificata.
 */
export function mean(this: Matrix, dim: 1 | 2 = 1): Matrix {
    const s = this.sum(dim);
    const divisor = dim === 1 ? this.rows : this.cols;
    
    divideInPlace(s.data, divisor);
    
    return s;
}

// ============================================================================
// LOGICA PRIVATA (Helper)
// ============================================================================

/**
 * Riduzione lungo le colonne (Verticale)
 * Strategia: Cache-friendly. Iteriamo per righe e accumuliamo nel buffer di output.
 */
function sumColumns(A: Matrix): Matrix {
    const out = new (A.constructor as any)(1, A.cols);
    const Adat = A.data, Odat = out.data;
    const { rows, cols } = A;

    for (let i = 0; i < rows; i++) {
        const offset = i * cols;
        let j = 0;
        
        // Loop Unrolling x4
        for (; j <= cols - 4; j += 4) {
            Odat[j]     += Adat[offset + j];
            Odat[j + 1] += Adat[offset + j + 1];
            Odat[j + 2] += Adat[offset + j + 2];
            Odat[j + 3] += Adat[offset + j + 3];
        }
        // Residuo
        for (; j < cols; j++) Odat[j] += Adat[offset + j];
    }
    return out;
}

/**
 * Riduzione lungo le righe (Orizzontale)
 * Strategia: Accumulatore locale per riga per minimizzare gli accessi in scrittura al buffer.
 */
function sumRows(A: Matrix): Matrix {
    const out = new (A.constructor as any)(A.rows, 1);
    const Adat = A.data, Odat = out.data;
    const { rows, cols } = A;

    for (let i = 0; i < rows; i++) {
        const offset = i * cols;
        let s = 0;
        let j = 0;

        // Loop Unrolling x4 (Accumulo in registro locale)
        for (; j <= cols - 4; j += 4) {
            s += Adat[offset + j]     + Adat[offset + j + 1] + 
                 Adat[offset + j + 2] + Adat[offset + j + 3];
        }
        // Residuo
        for (; j < cols; j++) s += Adat[offset + j];
        
        Odat[i] = s;
    }
    return out;
}

/**
 * Divisione in-place del buffer per il calcolo della media.
 */
function divideInPlace(data: Float64Array, divisor: number): void {
    const len = data.length;
    let i = 0;
    
    // Loop Unrolling x4
    for (; i <= len - 4; i += 4) {
        data[i]     /= divisor;
        data[i + 1] /= divisor;
        data[i + 2] /= divisor;
        data[i + 3] /= divisor;
    }
    // Residuo
    for (; i < len; i++) data[i] /= divisor;
}