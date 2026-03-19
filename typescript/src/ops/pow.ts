import  { Matrix }  from "..";

/**
 * Matrice elevata a potenza (mpower). Solo per matrici quadrate.
 */
export function pow(this: Matrix, exp: number): Matrix {
    if (this.rows !== this.cols) {
        throw new Error("Matrix power is only defined for square matrices.");
    }
    if (!Number.isInteger(exp) || exp < 0) {
        throw new Error("Only non-negative integer exponents are supported currently.");
    }

    if (exp === 0) return (this.constructor as any).eye(this.rows);
    if (exp === 1) return this.clone(); // Supponendo esista un metodo copy

    let res = (this.constructor as any).eye(this.rows);
    let base: Matrix = this;

    while (exp > 0) {
        if (exp % 2 === 1) {
            res = res.mul(base);
        }
        base = base.mul(base);
        exp = Math.floor(exp / 2);
    }
    return res;
}