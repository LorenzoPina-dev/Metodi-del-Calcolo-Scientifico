import { Matrix } from "..";
import { lup } from "../decomposition";
import { solveLowerTriangular, solveUpperTriangular } from "./triangular";

export function solve(A: Matrix, b: Matrix): Matrix {
    const { L, U, P } = lup(A);
    const n = A.rows;
    const Pb = new Matrix(n, b.cols);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < b.cols; j++)
            Pb.set(i, j, b.get(P[i], j));
    const y = solveLowerTriangular(L, Pb);
    return solveUpperTriangular(U, y);
}