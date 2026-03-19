import { Matrix } from "..";

/**
 * Motore interno per funzioni che mappano un numero in un altro.
 * Applica la funzione 'fn' a ogni elemento della matrice.
 */
function applyUnary(A: Matrix, fn: (x: number) => number): Matrix {
    const out = new Matrix(A.rows, A.cols);
    const Adat = A.data;
    const Cdat = out.data;
    const len = Adat.length;

    let i = 0;
    // Loop Unrolling (fattore 4) per massimizzare il throughput della CPU
    for (; i <= len - 4; i += 4) {
        Cdat[i] = fn(Adat[i]);
        Cdat[i + 1] = fn(Adat[i + 1]);
        Cdat[i + 2] = fn(Adat[i + 2]);
        Cdat[i + 3] = fn(Adat[i + 3]);
    }

    // Gestione degli elementi rimanenti
    for (; i < len; i++) {
        Cdat[i] = fn(Adat[i]);
    }

    return out;
}

/**
 * ABS - Valore assoluto di ogni elemento (MATLAB: abs(A))
 */
export function abs(this: Matrix): Matrix {
    return applyUnary(this, Math.abs);
}

/**
 * SQRT - Radice quadrata di ogni elemento (MATLAB: sqrt(A))
 */
export function sqrt(this: Matrix): Matrix {
    return applyUnary(this, Math.sqrt);
}

/**
 * SIN - Seno di ogni elemento (MATLAB: sin(A))
 */
export function sin(this: Matrix): Matrix {
    return applyUnary(this, Math.sin);
}

/**
 * COS - Coseno di ogni elemento (MATLAB: cos(A))
 */
export function cos(this: Matrix): Matrix {
    return applyUnary(this, Math.cos);
}

/**
 * TAN - Tangente di ogni elemento (MATLAB: tan(A))
 */
export function tan(this: Matrix): Matrix {
    return applyUnary(this, Math.tan);
}

/** * Traccia della matrice (somma elementi diagonale). 
 * Ottimizzata per saltare i dati non necessari.
 */
export function trace(this: Matrix): number {
    const n = Math.min(this.rows, this.cols);
    const Adat = this.data;
    const stride = this.cols + 1; // Distanza tra elementi sulla diagonale
    let t = 0;
    for (let i = 0; i < n; i++) {
        t += Adat[i * stride];
    }
    return t;
}

/**
 * Esponenziale element-wise (e^x).
 */
export function exp(this: Matrix): Matrix {
    const out = new (this.constructor as any)(this.rows, this.cols);
    const Adat = this.data, Odat = out.data;
    const len = Adat.length;
    let i = 0;
    for (; i <= len - 4; i += 4) {
        Odat[i] = Math.exp(Adat[i]);
        Odat[i + 1] = Math.exp(Adat[i + 1]);
        Odat[i + 2] = Math.exp(Adat[i + 2]);
        Odat[i + 3] = Math.exp(Adat[i + 3]);
    }
    for (; i < len; i++) Odat[i] = Math.exp(Adat[i]);
    return out;
}


/**
 * Arrotonda gli elementi della matrice all'intero più vicino.
 */
export function round(this: Matrix): Matrix {
    const out = new (this.constructor as any)(this.rows, this.cols);
    
    // Deleghiamo la logica pesante all'helper ottimizzato
    processRound(this.data, out.data);
    
    return out;
}

/**
 * Arrotonda verso il basso (pavimento).
 */
export function floor(this: Matrix): Matrix {
    const out = new (this.constructor as any)(this.rows, this.cols);
    processFloor(this.data, out.data);
    return out;
}

/**
 * Arrotonda verso l'alto (soffitto).
 */
export function ceil(this: Matrix): Matrix {
    const out = new (this.constructor as any)(this.rows, this.cols);
    processCeil(this.data, out.data);
    return out;
}

// ============================================================================
// LOGICA PRIVATA (Helper ottimizzati)
// ============================================================================

function processRound(input: Float64Array, output: Float64Array): void {
    const len = input.length;
    let i = 0;

    // Loop Unrolling x4: Riduce i salti del contatore e migliora il throughput
    for (; i <= len - 4; i += 4) {
        output[i]     = Math.round(input[i]);
        output[i + 1] = Math.round(input[i + 1]);
        output[i + 2] = Math.round(input[i + 2]);
        output[i + 3] = Math.round(input[i + 3]);
    }

    // Gestione dei residui (elementi rimanenti se len non è multiplo di 4)
    for (; i < len; i++) {
        output[i] = Math.round(input[i]);
    }
}

function processFloor(input: Float64Array, output: Float64Array): void {
    const len = input.length;
    let i = 0;
    for (; i <= len - 4; i += 4) {
        output[i]     = Math.floor(input[i]);
        output[i + 1] = Math.floor(input[i + 1]);
        output[i + 2] = Math.floor(input[i + 2]);
        output[i + 3] = Math.floor(input[i + 3]);
    }
    for (; i < len; i++) output[i] = Math.floor(input[i]);
}

function processCeil(input: Float64Array, output: Float64Array): void {
    const len = input.length;
    let i = 0;
    for (; i <= len - 4; i += 4) {
        output[i]     = Math.ceil(input[i]);
        output[i + 1] = Math.ceil(input[i + 1]);
        output[i + 2] = Math.ceil(input[i + 2]);
        output[i + 3] = Math.ceil(input[i + 3]);
    }
    for (; i < len; i++) output[i] = Math.ceil(input[i]);
}