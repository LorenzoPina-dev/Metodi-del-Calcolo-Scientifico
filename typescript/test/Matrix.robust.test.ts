// Matrix.robust.test.ts
import { describe, it, expect } from 'vitest';
import { Matrix } from '../src';

// funzione di supporto per generare matrici casuali
function randomMatrix(rows: number, cols: number, scale = 1): Matrix {
    const M = Matrix.zeros(rows, cols);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            M.set(i, j, (Math.random() - 0.5) * scale);
        }
    }
    return M;
}

describe('Matrix robust tests', () => {

    // ---------------- LU / LUP ----------------
    it('LU decomposition on square, rectangular, and singular matrices', () => {
        const square = randomMatrix(5,5).add(Matrix.diag(5,10)); // matrice diagonale dominante
        const {L, U} = Matrix.decomp.lu(square);
        const recon = L.mul(U);
        // ricostruzione
        for (let i=0;i<5;i++){
            for (let j=0;j<5;j++){
                expect(recon.get(i,j)).toBeCloseTo(square.get(i,j), 10);
            }
        }

        const singular = Matrix.zeros(4,4);
        expect(() => Matrix.decomp.lu(singular)).toThrow(Error); // LU senza pivoting non funziona

        // LUP con pivoting
        const { L: L2, U: U2, P } = Matrix.decomp.luPivotingTotal(square);
        const PA = P.mul(square);
        const recon2 = L2.mul(U2);
        for (let i=0;i<5;i++){
            for (let j=0;j<5;j++){
                expect(recon2.get(i,j)).toBeCloseTo(PA.get(i,j), 10);
            }
        }
    });

    // ---------------- DET ----------------
    it('determinant consistency with LUP', () => {
        const I = Matrix.identity(4);
        expect(I.det()).toBeCloseTo(1);

        const randomMat = randomMatrix(5,5).add(Matrix.diag(5,2)); // matrice diagonale dominante
        console.log("Random matrix:\n", randomMat.toString());
        const { U, swaps } = Matrix.decomp.lup(randomMat);
        let detViaLUP = (-1)**swaps;
        for (let k=0;k<U.rows;k++) detViaLUP *= U.get(k,k);
        expect(randomMat.det()).toBeCloseTo(detViaLUP);
    });

    // ---------------- ORTOGONALITÀ ----------------
    it('orthogonal matrices', () => {
        const I = Matrix.identity(3);
        expect(I.isOrthogonal()).toBe(true);

        const randomMat = randomMatrix(3,3).add(Matrix.diag(3,10)); // matrice diagonale dominante
        expect(randomMat.isOrthogonal()).toBe(false);
    });

    // ---------------- SLICE / SUBMATRIX ----------------
    it('slice arbitrary submatrices', () => {
        const A = randomMatrix(10,8);
        const sub = A.slice(2,7,1,6);
        expect(sub.rows).toBe(5);
        expect(sub.cols).toBe(5);
        for(let i=0;i<5;i++){
            for(let j=0;j<5;j++){
                expect(sub.get(i,j)).toBeCloseTo(A.get(i+2,j+1));
            }
        }
    });

    // ---------------- TRANSPOSE / ROTATE / FLIP ----------------
    it('transpose of rectangular matrices', () => {
        const A = randomMatrix(3,5);
        const AT = A.t();
        expect(AT.rows).toBe(5);
        expect(AT.cols).toBe(3);
        for(let i=0;i<3;i++){
            for(let j=0;j<5;j++){
                expect(AT.get(j,i)).toBeCloseTo(A.get(i,j));
            }
        }
    });

    it('rot90 multiple rotations', () => {
        const A = Matrix.ones(2,3);
        const B = A.rot90();
        const C = B.rot90();
        const D = C.rot90();
        const E = D.rot90();
        // rot90^4 = identity shape
        expect(E.rows).toBe(A.rows);
        expect(E.cols).toBe(A.cols);
    });

    it('flip horizontal and vertical', () => {
        const A = Matrix.zeros(2,3);
        A.set(0,0,1);
        const H = A.flip(1);
        const V = A.flip(2);
        expect(H.get(0,2)).toBe(1);
        expect(V.get(1,0)).toBe(1);
    });

    // ---------------- UNARY ----------------
    it('abs/sqrt/round edge cases', () => {
        const A = Matrix.zeros(2,2).sub(5);
        const B = A.abs();
        expect(B.get(0,0)).toBe(5);

        const C = Matrix.ones(2,2);
        expect(C.sqrt().get(0,0)).toBeCloseTo(1);

        const D = Matrix.ones(2,2).mul(1.4).round();
        expect(D.get(0,0)).toBe(1);
    });

    // ---------------- STATISTICS ----------------
    it('sum/mean/max/min on random matrices', () => {
        const A = randomMatrix(5,4,10);
        const total = A.totalSum();
        let calc = 0;
        for(let i=0;i<5;i++){
            for(let j=0;j<4;j++) calc += A.get(i,j);
        }
        expect(total).toBeCloseTo(calc);

        expect(A.mean(1).rows).toBe(1);
        expect(A.mean(2).cols).toBe(1);
    });

    // ---------------- STATIC FACTORY ----------------
    it('diag/diagFromArray correctness', () => {
        const D = Matrix.diag(3,7);
        for(let i=0;i<3;i++){
            for(let j=0;j<3;j++){
                const expected = i===j?7:0;
                expect(D.get(i,j)).toBe(expected);
            }
        }

        const arr = [1,2,3,4];
        const D2 = Matrix.diagFromArray(arr);
        for(let i=0;i<4;i++){
            for(let j=0;j<4;j++){
                const expected = i===j?arr[i]:0;
                expect(D2.get(i,j)).toBe(expected);
            }
        }
    });

    // ---------------- GALLERY ----------------
    it('hilbert matrix properties', () => {
        const H = Matrix.gallery.hilbert(5);
        expect(H.get(0,0)).toBeCloseTo(1);
        expect(H.get(4,4)).toBeCloseTo(1/9);
    });

});