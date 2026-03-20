import * as fs from "fs";
import * as path from "path";

import { Matrix } from "../src";
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
    {
        throw new Error("Shape mismatch");
    }

    for (let i = 0; i < A.rows; i++) {
        for (let j = 0; j < A.cols; j++) {
            const diff = Math.abs(A.get(i, j) - B.get(i, j));
            if (diff > tol) {
                console.log(`Mismatch at (${i},${j}): ${A.get(i, j)} vs ${B.get(i, j)}, diff = ${diff}`);
                if(A.rows <= 10 && A.cols <= 10) // print matrices only if small
                {console.log("Generated matrix:");
                console.log(A.toString());
                console.log("MATLAB matrix:");
                console.log(B.toString());
                }
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
            return Matrix.gallery.hilbert(params.n);
        case "pascal":
            return Matrix.gallery.pascal(params.n);
        case "magic":
            return Matrix.gallery.magic(params.n);
        case "lehmer":
            return Matrix.gallery.lehmer(params.n);
        case "grcar":
            return Matrix.gallery.grcar(params.n);
        case "fiedler":
            return Matrix.gallery.fiedler(params.v);
        case "circul":
            return Matrix.gallery.circul(params.c);
        case "tridiag":
            return Matrix.gallery.tridiag(params.a, params.b, params.c);
        //case "smoke":!!!!!!!!!!!!!!!!!!!!!!!!!!COMPLESSI!!!!!!!!!!!!!!!!!!!!!!
        //    return Matrix.gallery.smoke(params.n);
        case "dorr":
            return Matrix.gallery.dorr(params.n);
        case "hanowa":
            return Matrix.gallery.hanowa(params.n);
        case "neumann":
            return Matrix.gallery.neumann(params.n);
        case "cauchy":
            return Matrix.gallery.cauchy(params.x, params.y);
        case "binomial":
            return Matrix.gallery.binomial(params.n);
        case "randsvd":
            return Matrix.gallery.randsvd(params.n);
        case "toeplitz":
            return Matrix.toeplitz(params.c, params.r);
        case "frank":
            return Matrix.gallery.frank(params.n);
        case "invhess":
            return Matrix.gallery.invhess(params.n);
        case "kahan":
            return Matrix.gallery.kahan(params.n);
        case "wathen":
            return Matrix.gallery.wathen(params.nx,params.ny);
        case "wilkinson":
            return Matrix.gallery.wilkinson(params.n);  
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
            if(name === "wathen") {
                console.warn("⚠ Skipping wathen test due to random density (rho) in assembly");
                continue;
            }
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