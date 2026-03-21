// decomposition/index.ts
export * from "./cholesky";
export * from "./lu";
export * from "./lu_total";
export * from "./lup";
export * from "./qr";
export * from "./tril";
export * from "./triu";
export * from "./dlu";

// ---- Alias backward-compatible ----
// luPivoting: come lup ma restituisce P come Matrix (matrice di permutazione)
import { lup } from "./lup";
import { INumeric } from "../type";
import { Matrix } from "..";

export function luPivoting<T extends INumeric<T>>(
    A: Matrix<T>
): { L: Matrix<T>; U: Matrix<T>; P: Matrix<T>; swaps: number } {
    const { L, U, P: perm, swaps } = lup(A);
    const n = A.rows;
    const Pm = A.like(n, n);
    const pd = (Pm as any).data;
    for (let i = 0; i < n; i++) pd[i * n + perm[i]] = A.one;
    return { L, U, P: Pm, swaps };
}
