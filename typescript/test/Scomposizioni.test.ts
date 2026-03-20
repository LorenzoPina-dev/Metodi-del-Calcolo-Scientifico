// Scomposizioni.test.ts
import { Matrix } from "../src"; // Assumi che Matrix sia la tua classe
import { describe, it, expect, beforeAll } from 'vitest';

describe("Test sui metodi di scomposizione", () => {

  const tol = 1e-10;

  // Matrici di test
  const A_square = new Matrix(2,2,Float64Array.from([2, 3, 3, 6]));
  const A_spd = new Matrix(2,2,Float64Array.from([4, 2, 2, 3])); // simmetrica definita positiva
  const A_rect = new Matrix(3,2,Float64Array.from([1, 2, 3, 4, 5, 6])); // rettangolare
  const b = new Matrix(2,1,Float64Array.from([10, 12]));

  // Helper: verifica che A ≈ ricostruzione
  function testReconstruction(original: Matrix, reconstructed: Matrix) {
      expect(original.equals(reconstructed, tol)).toBe(true);
  }

  // ============================================
  // LU decomposition
  // ============================================
  describe("LU", () => {
    it("ricostruisce la matrice originale correttamente", () => {
      const { L, U } = Matrix.decomp.lu(A_square);
      const A_rec = L.mul(U);
      testReconstruction(A_square, A_rec);
    });

    it("L ha diagonale unitaria", () => {
      const { L } = Matrix.decomp.lu(A_square);
      for (let i = 0; i < L.rows; i++) {
        expect(Math.abs(L.get(i, i) - 1)).toBeLessThan(tol);
      }
    });

    it("dovrebbe fallire su matrice non invertibile", () => {
      const singular = new Matrix(2,2,Float64Array.from([0, 2, 1, 4]));
      expect(() => Matrix.decomp.lu(singular)).toThrow();
    });
  });

  // ============================================
  // LUP decomposition
  // ============================================
  describe("LUP", () => {
    it("ricostruisce la matrice originale correttamente con permutazione", () => {
      const { L, U, P } = Matrix.decomp.luPivoting(A_square);
      const PA = P.mul(A_square);
      const LU = L.mul(U);
      testReconstruction(PA, LU);
    });

    it("funziona anche su matrici che LU fallirebbe", () => {
      const singularRowSwap = new Matrix(2,2,Float64Array.from([0, 2, 1, 3]));
      const { L, U, P } = Matrix.decomp.luPivoting(singularRowSwap);
      const PA = P.mul(singularRowSwap);
      const LU = L.mul(U);
      console.log("PA:\n", PA.toString());
      console.log("LU:\n", LU.toString());
      testReconstruction(PA, LU);
    });
  });

  // ============================================
  // Cholesky decomposition
  // ============================================
  describe("Cholesky", () => {
    it("ricostruisce la matrice SPD correttamente", () => {
      const {L} = Matrix.decomp.cholesky(A_spd);
      const LLt = L.mul(L.t());
      testReconstruction(A_spd, LLt);
    });

    it("fallisce su matrice non SPD", () => {
        const nonSPD= new Matrix(2,2,Float64Array.from([1, 2, 2, 1]));
        expect(() => Matrix.decomp.cholesky(nonSPD)).toThrow();
    });
  });

  // ============================================
  // QR decomposition
  // ============================================
  describe("QR", () => {
    it("ricostruisce la matrice originale correttamente", () => {
      const { Q, R } = Matrix.decomp.qr(A_rect);
      const QR = Q.mul(R);
      testReconstruction(A_rect, QR);
    });

   it("Q è ortogonale", () => {
        const { Q } = Matrix.decomp.qr(A_rect);

        const QtQ = Q.t().mul(Q);
        const I = Matrix.identity(Q.cols); // <-- FIX
        console.log("QtQ:\n", QtQ.toString());
        console.log("I:\n", I.toString());

        testReconstruction(I, QtQ);
        });
    });
});