import * as fs from "fs";
import * as path from "path";

import * as M from "../src/init";
import { Matrix } from "../src/core";
import { zeros } from "../src/init";

/* ---------- LOAD JSON ---------- */
function loadJSON(file: string): { matrix: Matrix; params: any } {
    const text = fs.readFileSync(file, "utf-8");
    const obj = JSON.parse(text);

    let A: Matrix;
    if ("matrix" in obj) {
        const data = obj.matrix as number[][];
        const m = data.length;
        const n = data[0].length;
        A = zeros(m, n);
        for (let i = 0; i < m; i++)
            for (let j = 0; j < n; j++)
                A.set(i, j, data[i][j]);
    } else if ("matrix_real" in obj && "matrix_imag" in obj) {
        // complex matrix: only real part for comparison
        const data = obj.matrix_real as number[][];
        const m = data.length;
        const n = data[0].length;
        A = zeros(m, n);
        for (let i = 0; i < m; i++)
            for (let j = 0; j < n; j++)
                A.set(i, j, data[i][j]);
    } else {
        throw new Error("Invalid JSON structure");
    }

    return { matrix: A, params: obj.params };
}

/* ---------- ASSERT ---------- */
function assertClose(A: Matrix, B: Matrix, tol = 1e-8) {
    if (A.rows !== B.rows || A.cols !== B.cols)
        throw new Error("Shape mismatch");

    for (let i = 0; i < A.rows; i++) {
        for (let j = 0; j < A.cols; j++) {
            const diff = Math.abs(A.get(i, j) - B.get(i, j));
            if (diff > tol) {
                console.log(`Mismatch at (${i},${j}): ${A.get(i, j)} vs ${B.get(i, j)}, diff = ${diff}`);
                console.log("Generated matrix:");
                console.log(A.toString());
                console.log("MATLAB matrix:");
                console.log(B.toString());
                throw new Error(`Mismatch at (${i},${j}): ${A.get(i,j)} vs ${B.get(i,j)}`);
            }
        }
    }
}

/* ---------- PROPERTIES ---------- */
function checkProperties(name: string, A: Matrix) {
    if (name === "hilb" && !A.isSymmetric())
        throw new Error("Hilbert non simmetrica");

    if (name === "toeplitz") {
        for (let i = 1; i < A.rows; i++)
            for (let j = 1; j < A.cols; j++)
                if (A.get(i, j) !== A.get(i - 1, j - 1))
                    throw new Error("Toeplitz property violated");
    }

    if (name === "grcar") {
        for (let i = 0; i < A.rows; i++)
            if (A.get(i, i) !== 1)
                throw new Error("Grcar diag error");
    }
}

/* ---------- GENERATOR MATCH ---------- */
function generateMatrix(name: string, params: any): Matrix {
    switch (name) {
        case "hilb":
            return M.hilbert(params.n);
        case "pascal":
            return M.pascal(params.n);
        case "magic":
            return M.magic(params.n);
        case "lehmer":
            return M.lehmer(params.n);
        case "grcar":
            return M.grcar(params.n);
        case "toeplitz":
            return M.toeplitz(params.c, params.r);
        case "fiedler":
            return M.fiedler(params.v);
        case "circul":
            return M.circul(params.c);
        case "tridiag":
            return M.tridiag(params.a, params.b, params.c);
        case "smoke":
            return M.smoke(params.n);
        case "dorr":
            return M.dorr(params.n);
        case "hanowa":
            return M.hanowa(params.n);
        case "neumann":
            return M.neumann(params.n);
        case "cauchy":
            return M.cauchy(params.x, params.y);
        case "binomial":
            return M.binomial(params.n, params.p);
        case "randsvd":
            return M.randsvd(params.n);
        case "toeplitz":
            return M.toeplitz(params.c, params.r);
        case "frank":
            return M.frank(params.n);
        case "invhess":
            return M.invhess(params.n);
        case "kahan":
            return M.kahan(params.n);

        default:
            throw new Error(`Not implemented: ${name}`);
    }
}

/* ---------- MAIN TEST ---------- */
function runTests(baseDir: string) {
    const matrixTypes = fs.readdirSync(baseDir);

    for (const name of matrixTypes) {
        const dir = path.join(baseDir, name);
        const files = fs.readdirSync(dir).filter(f => f.endsWith(".json"));

        for (const file of files) {
            const filePath = path.join(dir, file);
            const { matrix: matlabMatrix, params } = loadJSON(filePath);
            const generated = generateMatrix(name, params);

            console.log(`Testing ${name} ${file}`);

            assertClose(generated, matlabMatrix);
            checkProperties(name, generated);
        }
    }

    console.log("✔ Tutti i test passati");
}

/* ---------- RUN ---------- */
runTests("truth_matrices");