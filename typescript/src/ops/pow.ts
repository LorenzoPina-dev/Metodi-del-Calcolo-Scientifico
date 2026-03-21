// ops/pow.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/** Potenza matriciale (mpower): A^exp, solo quadrate e esponente intero ≥ 0. */
export function pow<T extends INumeric<T>>(this: Matrix<T>, exp: number): Matrix<T> {
    if (this.rows !== this.cols) throw new Error("pow: solo matrici quadrate.");
    if (!Number.isInteger(exp) || exp < 0) throw new Error("pow: solo esponenti interi non negativi.");
    if (exp === 0) return this.likeIdentity(this.rows);
    if (exp === 1) return this.clone();

    // Esponenziazione rapida (binary exponentiation)
    let res = this.likeIdentity(this.rows);
    let base = this.clone();
    let e = exp;
    while (e > 0) {
        if (e % 2 === 1) res = res.mul(base);
        base = base.mul(base);
        e = Math.floor(e / 2);
    }
    return res;
}
