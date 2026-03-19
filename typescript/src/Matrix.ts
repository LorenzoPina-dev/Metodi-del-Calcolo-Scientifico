// Matrix.ts

import { MatrixBase } from "./core/MatrixBase";

// --- OPS ---
import * as addOps from "./ops/add";
import * as subOps from "./ops/subtract";
import * as mulOps from "./ops/multiply";
import * as dotOps from "./ops/dotOps";
import * as statOps from "./ops/statistics";
import * as unaryOps from "./ops/unary";
import * as transOps from "./ops/transform";

// --- GALLERY ---
import * as gallery from "./init/known";
// --- SOLVER ---
import * as solver from "./solver";
// --- DECOMP ---
import * as decomp from "./decomposition";

import { det } from "./ops/det";
import { norm } from "./ops/norm";
import { pow } from "./ops/pow";


import { smartInverse } from "./algoritm/inverse";

// --- INIT ---
import * as init from "./init/init";
import { hankel, random, sparse, toeplitz, vander } from "./init";
import * as propOps from "./ops/hasProperty";
import { tril, triu } from "./decomposition";

export class Matrix extends MatrixBase {

    totalSum(): number {
        return addOps.totalSum.call(this);
    }

    // -------- ARITMETICA --------

    add(B: Matrix | number) {
        return addOps.add.call(this, B);
    }

    sub(B: Matrix | number) {
        return subOps.subtract.call(this, B);
    }

    mul(B: Matrix | number) {
        return mulOps.multiply.call(this, B);
    }

    pow(exp: number) {
        return pow.call(this, exp);
    }

    // -------- DOT OPS --------
    dotMul(B: Matrix | number) {
        return dotOps.dotMultiply.call(this, B);
    }

    dotDiv(B: Matrix | number) {
        return dotOps.dotDivide.call(this, B);
    }

    dotPow(exp: number | Matrix) {
        return dotOps.dotPow.call(this, exp);
    }

    // -------- ALGEBRA --------
    det() {
        return det(this);
    }

    norm(type?: any) {
        return norm.call(this, type);
    }

    inv() {
        return smartInverse(this);
    }

    trace() {
        let t = 0;
        const n = Math.min(this.rows, this.cols);

        for (let i = 0; i < n; i++) {
            t += this.get(i, i);
        }

        return t;
    }

    // -------- STAT --------
    sum(dim: 1 | 2 = 1) {
        return statOps.sum.call(this, dim);
    }

    max(dim: 1 | 2 = 1) {
        return statOps.max.call(this, dim);
    }

    min(dim: 1 | 2 = 1) {
        return statOps.min.call(this, dim);
    }

    mean(dim: 1 | 2 = 1) {
        return statOps.mean.call(this, dim);
    }


    // -------- TRANSFORM --------
    t() {
        return transOps.transpose.call(this);
    }


    reshape(r: number, c: number) {
        return transOps.reshape.call(this, r, c);
    }

    repmat(r: number, c: number) {
        return transOps.repmat.call(this, r, c);
    }
    slice(rowStart: number, rowEnd: number, colStart: number, colEnd: number) {
        return transOps.slice.call(this, rowStart, rowEnd, colStart, colEnd);
    }

    // -------- UNARY --------
    abs() {
        return unaryOps.abs.call(this);
    }

    sqrt() {
        return unaryOps.sqrt.call(this);
    }

    round() {
        return unaryOps.round.call(this);
    }

    flip(dim: 1 | 2 = 1) {
        return transOps.flip.call(this, dim);
    }
    rot90(k: number = 1) {
        return transOps.rot90.call(this, k);
    }
    inverse() {
        return this.inv();
    }

    // -------- PROPERTIES --------
    isSquare() { return propOps.isSquare.call(this); }
    isSymmetric(tol?: number) { return propOps.isSymmetric.call(this, tol); }
    isUpperTriangular(tol?: number) { return propOps.isUpperTriangular.call(this, tol); }
    isLowerTriangular(tol?: number) { return propOps.isLowerTriangular.call(this, tol); }
    isDiagonal(tol?: number) { return propOps.isDiagonal.call(this, tol); }
    isIdentity(tol?: number) { return propOps.isIdentity.call(this, tol); }
    isOrthogonal(tol?: number) { return propOps.isOrthogonal.call(this, tol); }
    isZeroMatrix(tol?: number) { return propOps.isZeroMatrix.call(this, tol); }
    
    // -------- STATIC FACTORY --------
    static zeros = init.zeros;
    static ones = init.ones;
    static identity = init.identity;
    static diag = init.diag;
    static diagFromArray = init.diagFromArray;
    static random = random;
    static sparse = sparse;
    static toeplitz = toeplitz;
    static vander = vander;
    static hankel = hankel;
    static tril = tril;
    static triu = triu;

    static solver= solver;
    static gallery = gallery;
    static decomp = decomp;
}