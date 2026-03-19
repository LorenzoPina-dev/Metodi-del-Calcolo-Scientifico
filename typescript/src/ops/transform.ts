import { Matrix } from "..";

/**
 * Trasposizione: A.' in MATLAB.
 * Inverte righe con colonne.
 */
export function transpose(this: Matrix): Matrix {
    const out = new Matrix(this.cols, this.rows);
    const Adat = this.data, Cdat = out.data;
    const R = this.rows, C = this.cols;

    for (let i = 0; i < R; i++) {
        const iOff = i * C;
        for (let j = 0; j < C; j++) {
            Cdat[j * R + i] = Adat[iOff + j];
        }
    }
    return out;
}

/**
 * Reshape: Cambia le dimensioni mantenendo il numero totale di elementi.
 * Molto efficiente perché condivide il buffer di memoria.
 */
export function reshape(this: Matrix, r: number, c: number): Matrix {
    if (r * c !== this.rows * this.cols) {
        throw new Error("Reshape: total number of elements must not change.");
    }
    return new Matrix(r, c, this.data);
}

/**
 * Flip: Ribalta la matrice.
 * dim 1: Verticale (top-to-bottom) - MATLAB: flipud
 * dim 2: Orizzontale (left-to-right) - MATLAB: fliplr
 */
export function flip(this: Matrix, dim: 1 | 2 = 1): Matrix {
    const out = new Matrix(this.rows, this.cols);
    const Adat = this.data, Cdat = out.data;
    const R = this.rows, C = this.cols;

    if (dim === 1) { // Flip Verticale
        for (let i = 0; i < R; i++) {
            const targetRow = (R - 1 - i) * C;
            const sourceRow = i * C;
            Cdat.set(Adat.subarray(sourceRow, sourceRow + C), targetRow);
        }
    } else { // Flip Orizzontale
        for (let i = 0; i < R; i++) {
            const off = i * C;
            for (let j = 0; j < C; j++) {
                Cdat[off + (C - 1 - j)] = Adat[off + j];
            }
        }
    }
    return out;
}

/**
 * Rot90: Ruota la matrice di 90 gradi in senso antiorario per k volte.
 */
export function rot90(this: Matrix, k: number = 1): Matrix {
    k = ((k % 4) + 4) % 4; // Normalizza k tra 0 e 3
    if (k === 0) return this.clone() as Matrix;
    
    let res: Matrix;
    if (k === 1) { // 90° antiorario
        res = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                res.data[(this.cols - 1 - j) * this.rows + i] = this.data[i * this.cols + j];
            }
        }
    } else if (k === 2) { // 180°
        res = this.flip(1).flip(2);
    } else { // 270° antiorario (o 90° orario)
        res = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                res.data[j * this.rows + (this.rows - 1 - i)] = this.data[i * this.cols + j];
            }
        }
    }
    return res;
}

/**
 * Circshift: Scostamento circolare degli elementi.
 */
export function circshift(this: Matrix, rShift: number, cShift: number): Matrix {
    const out = new Matrix(this.rows, this.cols);
    const R = this.rows, C = this.cols;
    
    // Normalizzazione shift per evitare indici negativi o troppo grandi
    rShift = ((rShift % R) + R) % R;
    cShift = ((cShift % C) + C) % C;

    for (let i = 0; i < R; i++) {
        const newI = (i + rShift) % R;
        for (let j = 0; j < C; j++) {
            const newJ = (j + cShift) % C;
            out.data[newI * C + newJ] = this.data[i * C + j];
        }
    }
    return out;
}

/**
 * Repmat: Replica la matrice in un pattern r x c.
 */
export function repmat(this: Matrix, r: number, c: number): Matrix {
    const out = new Matrix(this.rows * r, this.cols * c);
    const R = this.rows, C = this.cols;
    
    for (let i = 0; i < r; i++) {
        for (let j = 0; j < c; j++) {
            // Posizionamento del blocco
            for (let rBlock = 0; rBlock < R; rBlock++) {
                const sourceOff = rBlock * C;
                const targetOff = (i * R + rBlock) * out.cols + (j * C);
                out.data.set(this.data.subarray(sourceOff, sourceOff + C), targetOff);
            }
        }
    }
    return out;
}

export function slice(this: Matrix, rowStart: number, rowEnd: number, colStart: number, colEnd: number): Matrix {
    const nRows = rowEnd - rowStart;
    const nCols = colEnd - colStart;

    if (nRows <= 0 || nCols <= 0) {
        throw new Error("Slice dimensions must be positive");
    }

    const result = new Matrix(nRows, nCols);

    const srcData = this.data;
    const dstData = result.data;

    let dstIndex = 0;

    for (let i = rowStart; i < rowEnd; i++) {
        const rowOffset = i * this.cols;
        for (let j = colStart; j < colEnd; j++) {
            dstData[dstIndex++] = srcData[rowOffset + j];
        }
    }

    return result;
}