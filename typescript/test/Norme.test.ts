// test/Norme.test.ts
import { describe, expect, it } from "vitest";
import { Matrix } from "../src";

describe("Test sulle norme", () => {

  const tol = 1e-10;

  // Costruzione con la vecchia API (Float64Array) — backward-compat
  const A = new Matrix(2, 2, Float64Array.from([
    1, -2,
    3, -4
  ]));

  it("Norma 1 corretta", () => {
    // max somma colonne: col1=|1|+|3|=4, col2=|-2|+|-4|=6
    expect(Math.abs(A.norm("1") - 6)).toBeLessThan(tol);
  });

  it("Norma infinito corretta", () => {
    // max somma righe: riga1=|1|+|-2|=3, riga2=|3|+|-4|=7
    expect(Math.abs(A.norm("Inf") - 7)).toBeLessThan(tol);
  });

  it("Norma Frobenius corretta", () => {
    // sqrt(1+4+9+16) = sqrt(30)
    expect(Math.abs(A.norm("Fro") - Math.sqrt(30))).toBeLessThan(tol);
  });

  it("Norme sono sempre >= 0", () => {
    expect(A.norm("1")).toBeGreaterThanOrEqual(0);
    expect(A.norm("Inf")).toBeGreaterThanOrEqual(0);
    expect(A.norm("Fro")).toBeGreaterThanOrEqual(0);
  });

  it("Norma della matrice nulla è zero", () => {
    const Z = new Matrix(2, 2);  // zero matrix (default)
    expect(Z.norm("1")).toBeCloseTo(0);
    expect(Z.norm("Inf")).toBeCloseTo(0);
    expect(Z.norm("Fro")).toBeCloseTo(0);
  });

  it("||cA|| = |c| * ||A||", () => {
    const c = -3;
    const B = A.mul(c);
    expect(B.norm("1")).toBeCloseTo(Math.abs(c) * A.norm("1"), 10);
    expect(B.norm("Inf")).toBeCloseTo(Math.abs(c) * A.norm("Inf"), 10);
    expect(B.norm("Fro")).toBeCloseTo(Math.abs(c) * A.norm("Fro"), 10);
  });

  it("||A + B|| <= ||A|| + ||B||", () => {
    const B = new Matrix(2, 2, Float64Array.from([2, 1, -1, 3]));
    const sum = A.add(B);
    expect(sum.norm("1")).toBeLessThanOrEqual(A.norm("1") + B.norm("1") + tol);
    expect(sum.norm("Inf")).toBeLessThanOrEqual(A.norm("Inf") + B.norm("Inf") + tol);
    expect(sum.norm("Fro")).toBeLessThanOrEqual(A.norm("Fro") + B.norm("Fro") + tol);
  });

  it("||AB|| <= ||A|| * ||B|| (norma 1 e infinito)", () => {
    const B = new Matrix(2, 2, Float64Array.from([1, 0, 0, 1]));
    const AB = A.mul(B);
    expect(AB.norm("1")).toBeLessThanOrEqual(A.norm("1") * B.norm("1") + tol);
    expect(AB.norm("Inf")).toBeLessThanOrEqual(A.norm("Inf") * B.norm("Inf") + tol);
  });
});
