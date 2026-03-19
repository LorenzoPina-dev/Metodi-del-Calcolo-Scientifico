import { Matrix } from "..";
import { zeros } from "./init";

export function toeplitz(c: number[], r?: number[]): Matrix {
    r = r ?? c;
    const T = zeros(c.length, r.length);

    for (let i = 0; i < c.length; i++)
        for (let j = 0; j < r.length; j++)
            T.set(i, j, j >= i ? r[j - i] : c[i - j]);

    return T;
}

export function hankel(c: number[], r?: number[]): Matrix {
    r = r ?? c;
    const H = zeros(c.length, r.length);

    for (let i = 0; i < c.length; i++)
        for (let j = 0; j < r.length; j++)
            H.set(i, j, i + j < c.length ? c[i + j] : r[i + j - c.length + 1]);

    return H;
}

export function vander(v: number[]): Matrix {
    const n = v.length;
    const V = zeros(n, n);

    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            V.set(i, j, Math.pow(v[i], n - j - 1));

    return V;
}