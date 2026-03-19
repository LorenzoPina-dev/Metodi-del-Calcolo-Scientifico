// Matrix.test.ts
import { describe, it, expect, beforeAll } from 'vitest';
import { Matrix } from '../src';

function measureTime(fn: () => any): number {
    const start = performance.now();
    fn();
    const end = performance.now();
    return end - start;
}

describe('Matrix core operations', () => {

    let A: Matrix;
    let B: Matrix;

    beforeAll(() => {
        A = Matrix.ones(100, 100);  // matrice di test
        B = Matrix.ones(100, 100).mul(2);
    });

    // ---------- ARITMETICA ----------
    it('add', () => {
        const time = measureTime(() => {
            const C = A.add(B);
            expect(C.get(0,0)).toBe(3);
            expect(C.rows).toBe(100);
        });
        console.log('add time (ms):', time);
    });

    it('sub', () => {
        const C = A.sub(B);
        expect(C.get(0,0)).toBe(-1);
    });

    it('mul', () => {
        const C = A.mul(B);
        expect(C.get(0,0)).toBe(2);
    });

    it('pow', () => {
        const C = A.pow(2);
        expect(C.get(0,0)).toBe(1);
    });

    // ---------- DOT OPS ----------
    it('dotMul', () => {
        const C = A.dotMul(B);
        expect(C.get(0,0)).toBe(2);
    });

    it('dotDiv', () => {
        const C = B.dotDiv(A);
        expect(C.get(0,0)).toBe(2);
    });

    it('dotPow', () => {
        const C = A.dotPow(3);
        expect(C.get(0,0)).toBe(1);
    });

    // ---------- ALGEBRA ----------
    it('trace', () => {
        expect(A.trace()).toBe(100);
    });

    it('inv', () => {
        const I = Matrix.identity(3);
        const invI = I.inv();
        expect(invI.isIdentity()).toBe(true);
    });

    it('totalSum', () => {
        expect(A.totalSum()).toBe(100*100);
    });

    it('det', () => {
        const I = Matrix.identity(3);
        expect(I.det()).toBeCloseTo(1);
    });

    // ---------- STATISTICHE ----------
    it('sum', () => {
        expect(A.sum(1).get(0,0)).toBe(100);
    });

    it('mean', () => {
        expect(A.mean(1).get(0,0)).toBe(1);
    });

    it('max/min', () => {
        expect(A.max().value.get(0,0)).toBe(1);
        expect(A.min().value.get(0,0)).toBe(1);
    });

    // ---------- PROPERTIES ----------
    it('isSquare', () => {
        expect(A.isSquare()).toBe(true);
    });

    it('isIdentity', () => {
        expect(Matrix.identity(3).isIdentity()).toBe(true);
    });

    it('isZeroMatrix', () => {
        expect(Matrix.zeros(3,3).isZeroMatrix()).toBe(true);
    });

    it('isOrthogonal', () => {
        const I = Matrix.identity(3);
        expect(I.isOrthogonal()).toBe(true);
    });

    // ---------- TRANSFORM ----------
    it('t/transpose', () => {
        const C = A.t();
        expect(C.rows).toBe(A.cols);
        expect(C.cols).toBe(A.rows);
        expect(A.t().rows).toBe(C.rows);
    });

    it('reshape', () => {
        const C = A.reshape(50, 200);
        expect(C.rows).toBe(50);
        expect(C.cols).toBe(200);
    });

    it('repmat', () => {
        const C = Matrix.ones(2,2).repmat(2,3);
        expect(C.rows).toBe(4);
        expect(C.cols).toBe(6);
    });

    it('slice', () => {
        const C = A.slice(0, 10, 0, 10);
        expect(C.rows).toBe(10);
        expect(C.cols).toBe(10);
        expect(C.get(0,0)).toBe(1);
    });

    // ---------- UNARY ----------
    it('abs/sqrt/round', () => {
        const D = Matrix.ones(2,2).mul(-2);
        expect(D.abs().get(0,0)).toBe(2);
        expect(Matrix.ones(2,2).sqrt().get(0,0)).toBe(1);
        expect(Matrix.ones(2,2).round().get(0,0)).toBe(1);
    });

    // ---------- ROTATE/FLIP ----------
    it('flip', () => {
        const C = Matrix.ones(2,3).flip(1);
        expect(C.rows).toBe(2);
        expect(C.cols).toBe(3);
    });

    it('rot90', () => {
        const C = Matrix.ones(2,3).rot90();
        expect(C.rows).toBe(3);
        expect(C.cols).toBe(2);
    });

    // ---------- STATIC FACTORY ----------
    it('zeros/ones/identity', () => {
        const Z = Matrix.zeros(3,3);
        const O = Matrix.ones(3,3);
        const I = Matrix.identity(3);
        expect(Z.isZeroMatrix()).toBe(true);
        expect(O.totalSum()).toBe(9);
        expect(I.isIdentity()).toBe(true);
    });

    it('diag/diagFromArray', () => {
        const D = Matrix.diag(3, 5);
        expect(D.get(0,0)).toBe(5);
        const D2 = Matrix.diagFromArray([1,2,3]);
        expect(D2.get(1,1)).toBe(2);
    });

    // ---------- GALLERY ----------
    it('hilbert', () => {
        const H = Matrix.gallery.hilbert(3);
        expect(H.get(0,0)).toBeCloseTo(1);
        expect(H.get(1,0)).toBeCloseTo(0.5);
    });

});