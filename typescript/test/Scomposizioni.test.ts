// test/Scomposizioni.test.ts
import { Matrix } from "../src";
import { describe, it, expect } from 'vitest';

describe("Test sui metodi di scomposizione", () => {

  const tol = 1e-10;

  const A_square = new Matrix(2, 2, Float64Array.from([2, 3, 3, 6]));
  const A_spd    = new Matrix(2, 2, Float64Array.from([4, 2, 2, 3]));
  const A_rect   = new Matrix(3, 2, Float64Array.from([1, 2, 3, 4, 5, 6]));

  function checkEqual(original: Matrix, reconstructed: Matrix) {
    expect(original.equals(reconstructed, tol)).toBe(true);
  }

  // ============================================
  // LU
  // ============================================
  describe("LU", () => {
    it("ricostruisce la matrice originale correttamente", () => {
      const { L, U } = Matrix.decomp.lu(A_square);
      checkEqual(A_square, L.mul(U));
    });

    it("L ha diagonale unitaria", () => {
      const { L } = Matrix.decomp.lu(A_square);
      for (let i = 0; i < L.rows; i++)
        expect(Math.abs(L.get(i, i).toNumber() - 1)).toBeLessThan(tol);
    });

    it("dovrebbe fallire su matrice con pivot nullo", () => {
      const singular = new Matrix(2, 2, Float64Array.from([0, 2, 1, 4]));
      expect(() => Matrix.decomp.lu(singular)).toThrow();
    });
  });

  // ============================================
  // LUP  (usa luPivoting che restituisce P come Matrix)
  // ============================================
  describe("LUP", () => {
    it("ricostruisce la matrice originale con permutazione", () => {
      const { L, U, P } = Matrix.decomp.luPivoting(A_square);
      // PA = P * A_square
      const PA = P.mul(A_square);
      checkEqual(PA, L.mul(U));
    });

    it("funziona anche su matrici che LU fallirebbe", () => {
      const m = new Matrix(2, 2, Float64Array.from([0, 2, 1, 3]));
      const { L, U, P } = Matrix.decomp.luPivoting(m);
      checkEqual(P.mul(m), L.mul(U));
    });
  });

  // ============================================
  // Cholesky
  // ============================================
  describe("Cholesky", () => {
    it("ricostruisce la matrice SPD correttamente", () => {
      const { L } = Matrix.decomp.cholesky(A_spd);
      checkEqual(A_spd, L.mul(L.t()));
    });

    it("fallisce su matrice non SPD", () => {
      const nonSPD = new Matrix(2, 2, Float64Array.from([1, 2, 2, 1]));
      expect(() => Matrix.decomp.cholesky(nonSPD)).toThrow();
    });
  });

  // ============================================
  // QR
  // ============================================
  describe("QR", () => {
    it("ricostruisce la matrice originale correttamente", () => {
      const { Q, R } = Matrix.decomp.qr(A_rect);
      checkEqual(A_rect, Q.mul(R));
    });

    it("Q è ortogonale (Q^T Q = I)", () => {
      const { Q } = Matrix.decomp.qr(A_rect);
      checkEqual(Matrix.identity(Q.cols), Q.t().mul(Q));
    });
  });
});
