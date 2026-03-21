// test/Matrix.robust.test.ts
import { describe, it, expect } from 'vitest';
import { Matrix } from '../src';

/** Genera una Matrix<Float64M> con valori casuali. */
function randomMatrix(rows: number, cols: number, scale = 1): Matrix {
    const M = Matrix.zeros(rows, cols);
    for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
            M.set(i, j, (Math.random() - 0.5) * scale);   // set accetta number
    return M;
}

describe('Matrix robust tests', () => {

    // ---- LU / LUP ----
    it('LU decomposition on square and singular matrices', () => {
        const square = randomMatrix(5, 5).add(Matrix.diag(5, 10));
        const { L, U } = Matrix.decomp.lu(square);
        const recon = L.mul(U);
        for (let i = 0; i < 5; i++)
            for (let j = 0; j < 5; j++)
                // get() restituisce Float64M; valueOf() permette la coercizione in toBeCloseTo
                expect(recon.get(i, j)).toBeCloseTo(square.get(i, j).value, 10);

        const singular = Matrix.zeros(4, 4);
        expect(() => Matrix.decomp.lu(singular)).toThrow(Error);

        // luPivoting restituisce P come Matrix
        const { L: L2, U: U2, P } = Matrix.decomp.luPivoting(square);
        const PA = P.mul(square);
        const recon2 = L2.mul(U2);
        for (let i = 0; i < 5; i++)
            for (let j = 0; j < 5; j++)
                expect(recon2.get(i, j)).toBeCloseTo(PA.get(i, j).value, 10);
    });

    // ---- DET ----
    it('determinant consistency with LUP', () => {
        const I = Matrix.identity(4);
        // det() restituisce Float64M; valueOf() → toBeCloseTo funziona
        expect(I.det()).toBeCloseTo(1);

        const randomMat = randomMatrix(5, 5).add(Matrix.diag(5, 2));
        const { U, swaps } = Matrix.decomp.lup(randomMat);
        let detViaLUP = (-1) ** swaps;
        for (let k = 0; k < U.rows; k++) detViaLUP *= U.get(k, k).value; // valueOf()
        expect(randomMat.det()).toBeCloseTo(detViaLUP);
    });

    // ---- ORTOGONALITÀ ----
    it('orthogonal matrices', () => {
        const I = Matrix.identity(3);
        expect(I.isOrthogonal()).toBe(true);

        const randomMat = randomMatrix(3, 3).add(Matrix.diag(3, 10));
        expect(randomMat.isOrthogonal()).toBe(false);
    });

    // ---- SLICE ----
    it('slice arbitrary submatrices', () => {
        const A = randomMatrix(10, 8);
        const sub = A.slice(2, 7, 1, 6);
        expect(sub.rows).toBe(5);
        expect(sub.cols).toBe(5);
        for (let i = 0; i < 5; i++)
            for (let j = 0; j < 5; j++)
                expect(sub.get(i, j)).toBeCloseTo(A.get(i + 2, j + 1).value);
    });

    // ---- TRANSPOSE ----
    it('transpose of rectangular matrices', () => {
        const A = randomMatrix(3, 5);
        const AT = A.t();
        expect(AT.rows).toBe(5);
        expect(AT.cols).toBe(3);
        for (let i = 0; i < 3; i++)
            for (let j = 0; j < 5; j++)
                expect(AT.get(j, i)).toBeCloseTo(A.get(i, j).value);
    });

    // ---- ROT90 / FLIP ----
    it('rot90 multiple rotations', () => {
        const A = Matrix.ones(2, 3);
        const E = A.rot90().rot90().rot90().rot90();
        expect(E.rows).toBe(A.rows);
        expect(E.cols).toBe(A.cols);
    });

    it('flip horizontal and vertical', () => {
        const A = Matrix.zeros(2, 3);
        A.set(0, 0, 1);                    // set(i,j,number) — backward-compat
        const H = A.flip(1);               // flipud
        const V = A.flip(2);               // fliplr

        expect(V.get(0, 2).toNumber()).toBe(1);   // fliplr: (0,0)→(0,2)
        expect(H.get(1, 0).toNumber()).toBe(1);   // flipud: (0,0)→(1,0)
    });

    // ---- UNARY ----
    it('abs/sqrt/round edge cases', () => {
        const A = Matrix.zeros(2, 2).sub(5);
        expect(A.abs().get(0, 0).toNumber()).toBe(5);
        expect(Matrix.ones(2, 2).sqrt().get(0, 0).toNumber()).toBeCloseTo(1);
        expect(Matrix.ones(2, 2).mul(1.4).round().get(0, 0).toNumber()).toBe(1);
    });

    // ---- STATISTICS ----
    it('sum/mean/max/min on random matrices', () => {
        const A = randomMatrix(5, 4, 10);
        const total = A.totalSum();
        let calc = 0;
        for (let i = 0; i < 5; i++)
            for (let j = 0; j < 4; j++) calc += A.get(i, j).value;  // valueOf()
        expect(total).toBeCloseTo(calc);

        expect(A.mean(1).rows).toBe(1);
        expect(A.mean(2).cols).toBe(1);
    });

    // ---- STATIC FACTORY ----
    it('diag/diagFromArray correctness', () => {
        const D = Matrix.diag(3, 7);
        for (let i = 0; i < 3; i++)
            for (let j = 0; j < 3; j++) {
                const expected = i === j ? 7 : 0;
                expect(D.get(i, j).toNumber()).toBe(expected);
            }

        const arr = [1, 2, 3, 4];
        const D2 = Matrix.diagFromArray(arr);
        for (let i = 0; i < 4; i++)
            for (let j = 0; j < 4; j++) {
                const expected = i === j ? arr[i] : 0;
                expect(D2.get(i, j).toNumber()).toBe(expected);
            }
    });

    // ---- GALLERY ----
    it('hilbert matrix properties', () => {
        const H = Matrix.gallery.hilbert(5);
        expect(H.get(0, 0).toNumber()).toBeCloseTo(1);
        expect(H.get(4, 4).toNumber()).toBeCloseTo(1 / 9);
    });
});
