// test/Matrix.test.ts
import { describe, it, expect, beforeAll } from 'vitest';
import { Matrix } from '../src';

describe('Matrix core operations', () => {

    let A: Matrix;
    let B: Matrix;

    beforeAll(() => {
        A = Matrix.ones(100, 100);
        B = Matrix.ones(100, 100).mul(2);
    });

    // ---------- ARITMETICA ----------
    it('add', () => {
        const C = A.add(B);
        // get() restituisce Float64M → .toNumber()
        expect(C.get(0, 0).toNumber()).toBe(3);
        expect(C.rows).toBe(100);
    });

    it('sub', () => {
        const C = A.sub(B);
        expect(C.get(0, 0).toNumber()).toBe(-1);
    });

    it('mul', () => {
        const C = A.mul(B);
        expect(C.get(0, 0).toNumber()).toBe(200);
    });

    it('pow', () => {
        const C = A.pow(2);
        expect(C.get(0, 0).toNumber()).toBe(100);
    });

    // ---------- DOT OPS ----------
    it('dotMul', () => {
        const C = A.dotMul(B);
        expect(C.get(0, 0).toNumber()).toBe(2);
    });

    it('dotDiv', () => {
        const C = B.dotDiv(A);
        expect(C.get(0, 0).toNumber()).toBe(2);
    });

    it('dotPow', () => {
        const C = A.dotPow(3);
        expect(C.get(0, 0).toNumber()).toBe(1);
    });

    // ---------- ALGEBRA ----------
    it('trace', () => {
        expect(A.trace().toNumber()).toBe(100);
    });

    it('inv', () => {
        const I = Matrix.identity(3);
        const invI = I.inv();
        expect(invI.isIdentity()).toBe(true);
    });

    it('totalSum', () => {
        expect(A.totalSum().toNumber()).toBe(100 * 100);
    });

    it('det', () => {
        const I = Matrix.identity(3);
        expect(I.det().toNumber()).toBeCloseTo(1);
    });

    // ---------- STATISTICHE ----------
    it('sum', () => {
        expect(A.sum(1).get(0, 0).toNumber()).toBe(100);
    });

    it('mean', () => {
        expect(A.mean(1).get(0, 0).toNumber()).toBe(1);
    });

    it('max/min', () => {
        expect(A.max().value.get(0, 0).toNumber()).toBe(1);
        expect(A.min().value.get(0, 0).toNumber()).toBe(1);
    });

    // ---------- PROPERTIES ----------
    it('isSquare', () => { expect(A.isSquare()).toBe(true); });
    it('isIdentity', () => { expect(Matrix.identity(3).isIdentity()).toBe(true); });
    it('isZeroMatrix', () => { expect(Matrix.zeros(3, 3).isZeroMatrix()).toBe(true); });
    it('isOrthogonal', () => { expect(Matrix.identity(3).isOrthogonal()).toBe(true); });

    // ---------- TRANSFORM ----------
    it('t/transpose', () => {
        const C = A.t();
        expect(C.rows).toBe(A.cols);
        expect(C.cols).toBe(A.rows);
    });

    it('reshape', () => {
        const C = A.reshape(50, 200);
        expect(C.rows).toBe(50);
        expect(C.cols).toBe(200);
    });

    it('repmat', () => {
        const C = Matrix.ones(2, 2).repmat(2, 3);
        expect(C.rows).toBe(4);
        expect(C.cols).toBe(6);
    });

    it('slice', () => {
        const C = A.slice(0, 10, 0, 10);
        expect(C.rows).toBe(10);
        expect(C.cols).toBe(10);
        expect(C.get(0, 0).toNumber()).toBe(1);
    });

    // ---------- UNARY ----------
    it('abs/sqrt/round', () => {
        const D = Matrix.ones(2, 2).mul(-2);
        expect(D.abs().get(0, 0).toNumber()).toBe(2);
        expect(Matrix.ones(2, 2).sqrt().get(0, 0).toNumber()).toBe(1);
        expect(Matrix.ones(2, 2).round().get(0, 0).toNumber()).toBe(1);
    });

    // ---------- ROTATE/FLIP ----------
    it('flip', () => {
        const C = Matrix.ones(2, 3).flip(1);
        expect(C.rows).toBe(2);
        expect(C.cols).toBe(3);
    });

    it('rot90', () => {
        const C = Matrix.ones(2, 3).rot90();
        expect(C.rows).toBe(3);
        expect(C.cols).toBe(2);
    });

    // ---------- STATIC FACTORY ----------
    it('zeros/ones/identity', () => {
        expect(Matrix.zeros(3, 3).isZeroMatrix()).toBe(true);
        expect(Matrix.ones(3, 3).totalSum().toNumber()).toBe(9);
        expect(Matrix.identity(3).isIdentity()).toBe(true);
    });

    it('diag/diagFromArray', () => {
        const D = Matrix.diag(3, 5);
        expect(D.get(0, 0).toNumber()).toBe(5);
        const D2 = Matrix.diagFromArray([1, 2, 3]);
        expect(D2.get(1, 1).toNumber()).toBe(2);
    });

    // ---------- GALLERY ----------
    it('hilbert', () => {
        const H = Matrix.gallery.hilbert(3);
        expect(H.get(0, 0).toNumber()).toBeCloseTo(1);
        expect(H.get(1, 0).toNumber()).toBeCloseTo(0.5);
    });
});

// ---------- Edge Cases ----------
describe('Matrix edge cases and additional operations', () => {

    it('should handle fromArray with empty array', () => {
        const M = Matrix.fromArray([]);
        expect(M.rows).toBe(0);
        expect(M.cols).toBe(0);
    });

    it('should handle fromArray with single element', () => {
        const M = Matrix.fromArray([[5]]);
        expect(M.rows).toBe(1);
        expect(M.cols).toBe(1);
        expect(M.get(0, 0).toNumber()).toBe(5);
    });

    it('should handle fromArray with rectangular matrix', () => {
        const M = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
        expect(M.rows).toBe(2);
        expect(M.cols).toBe(3);
        expect(M.get(0, 0).toNumber()).toBe(1);
        expect(M.get(1, 2).toNumber()).toBe(6);
    });

    it('equals with default tolerance', () => {
        const A = Matrix.fromArray([[1, 2], [3, 4]]);
        const B = Matrix.fromArray([[1, 2], [3, 4]]);
        expect(A.equals(B)).toBe(true);
    });

    it('equals with custom tolerance', () => {
        const A = Matrix.fromArray([[1.0, 2.0], [3.0, 4.0]]);
        const B = Matrix.fromArray([[1.001, 2.001], [3.001, 4.001]]);
        expect(A.equals(B, 0.01)).toBe(true);
        expect(A.equals(B, 0.0001)).toBe(false);
    });

    it('returns false for different dimensions', () => {
        const A = Matrix.fromArray([[1, 2], [3, 4]]);
        const B = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
        expect(A.equals(B)).toBe(false);
    });

    it('negate', () => {
        const M = Matrix.fromArray([[1, -2], [3, -4]]);
        const N = M.negate();
        expect(N.get(0, 0).toNumber()).toBe(-1);
        expect(N.get(0, 1).toNumber()).toBe(2);
        expect(N.get(1, 0).toNumber()).toBe(-3);
        expect(N.get(1, 1).toNumber()).toBe(4);
    });

    it('exp', () => {
        const M = Matrix.fromArray([[0, 1], [2, 3]]);
        const E = M.exp();
        expect(E.get(0, 0).toNumber()).toBeCloseTo(Math.exp(0));
        expect(E.get(0, 1).toNumber()).toBeCloseTo(Math.exp(1));
        expect(E.get(1, 0).toNumber()).toBeCloseTo(Math.exp(2));
        expect(E.get(1, 1).toNumber()).toBeCloseTo(Math.exp(3));
    });

    it('floor', () => {
        const M = Matrix.fromArray([[1.7, 2.3], [3.9, 4.1]]);
        const F = M.floor();
        expect(F.get(0, 0).toNumber()).toBe(1);
        expect(F.get(0, 1).toNumber()).toBe(2);
        expect(F.get(1, 0).toNumber()).toBe(3);
        expect(F.get(1, 1).toNumber()).toBe(4);
    });

    it('ceil', () => {
        const M = Matrix.fromArray([[1.1, 2.9], [3.2, 4.8]]);
        const C = M.ceil();
        expect(C.get(0, 0).toNumber()).toBe(2);
        expect(C.get(0, 1).toNumber()).toBe(3);
        expect(C.get(1, 0).toNumber()).toBe(4);
        expect(C.get(1, 1).toNumber()).toBe(5);
    });

    it('sin / cos / tan', () => {
        const M = Matrix.fromArray([[0, Math.PI / 2], [Math.PI, 3 * Math.PI / 2]]);

        const S = M.sin();
        expect(S.get(0, 0).toNumber()).toBeCloseTo(0);
        expect(S.get(0, 1).toNumber()).toBeCloseTo(1);

        const C = M.cos();
        expect(C.get(0, 0).toNumber()).toBeCloseTo(1);
        expect(C.get(0, 1).toNumber()).toBeCloseTo(0, 5);

        const T = Matrix.fromArray([[0, Math.PI / 4]]).tan();
        expect(T.get(0, 0).toNumber()).toBeCloseTo(0);
        expect(T.get(0, 1).toNumber()).toBeCloseTo(1, 5);
    });
});
