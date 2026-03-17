export class Matrix {
    data: Float64Array;
    rows: number;
    cols: number;

    constructor(rows: number, cols: number, data?: Float64Array) {
        this.rows = rows;
        this.cols = cols;
        this.data = data ?? new Float64Array(rows * cols);
    }

    // ---------------- INDEX ----------------
    private idx(i: number, j: number): number {
        return i * this.cols + j;
    }

    get(i: number, j: number): number {
        return this.data[this.idx(i, j)];
    }

    set(i: number, j: number, v: number): void {
        this.data[this.idx(i, j)] = v;
    }

    // ---------------- CONSTANTS ----------------
    static EPS = 1e-10;

    static isZero(x: number): boolean {
        return Math.abs(x) < Matrix.EPS;
    }

    // ---------------- FACTORY ----------------
    static zeros(r: number, c: number): Matrix {
        return new Matrix(r, c);
    }

    static identity(n: number): Matrix {
        return Matrix.diag(n, 1);
    }

    static ones(rows: number, cols: number): Matrix {
        const m = new Matrix(rows, cols);
        m.data.fill(1);
        return m;
    }

    static diag(n: number, k: number): Matrix {
        const m = Matrix.zeros(n, n);
        for (let i = 0; i < n; i++) m.data[i * n + i] = k;
        return m;
    }

    static diagFromArray(arr: number[]): Matrix {
        const n = arr.length;
        const m = Matrix.zeros(n, n);
        for (let i = 0; i < n; i++) m.data[i * n + i] = arr[i];
        return m;
    }

    // ---------------- COPY ----------------
    clone(): Matrix {
        return new Matrix(this.rows, this.cols, new Float64Array(this.data));
    }

    // ---------------- MATRICES ----------------
    triu(): Matrix {
        const result = Matrix.zeros(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = i; j < this.cols; j++) result.set(i, j, this.get(i, j));
        }
        return result;
    }

    tril(): Matrix {
        const result = Matrix.zeros(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j <= i; j++) result.set(i, j, this.get(i, j));
        }
        return result;
    }

    // ---------------- ALGEBRA LINEARE ----------------
    isSymmetric(): boolean {
        if (this.rows !== this.cols) return false;
        for (let i = 0; i < this.rows; i++) {
            for (let j = i + 1; j < this.cols; j++) {
                if (Math.abs(this.get(i, j) - this.get(j, i)) > Matrix.EPS) return false;
            }
        }
        return true;
    }

    add(B: Matrix): Matrix {
        const out = new Matrix(this.rows, this.cols);
        if (B.rows === 1 && B.cols === this.cols) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) + B.get(0, j));
        } else if (B.cols === 1 && B.rows === this.rows) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) + B.get(i, 0));
        } else if (B.rows === this.rows && B.cols === this.cols) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) + B.get(i, j));
        } else throw new Error("Incompatible shapes for add");
        return out;
    }

    subtract(B: Matrix): Matrix {
        const out = new Matrix(this.rows, this.cols);
        if (B.rows === 1 && B.cols === this.cols) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) - B.get(0, j));
        } else if (B.cols === 1 && B.rows === this.rows) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) - B.get(i, 0));
        } else if (B.rows === this.rows && B.cols === this.cols) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    out.set(i, j, this.get(i, j) - B.get(i, j));
        } else throw new Error("Incompatible shapes for subtract");
        return out;
    }

    multiply(B: Matrix): Matrix {
        if (this.cols !== B.rows) throw new Error("Dim mismatch");
        const out = new Matrix(this.rows, B.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let k = 0; k < this.cols; k++) {
                const aik = this.get(i, k);
                for (let j = 0; j < B.cols; j++) {
                    out.data[i * B.cols + j] += aik * B.get(k, j);
                }
            }
        }
        return out;
    }

    transpose(): Matrix {
        const out = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
                out.set(j, i, this.get(i, j));
        return out;
    }

    sum(axis: 0 | 1 = 0): Matrix {
        if (axis === 0) {
            const out = new Matrix(1, this.cols);
            for (let j = 0; j < this.cols; j++) {
                let s = 0;
                for (let i = 0; i < this.rows; i++) s += this.get(i, j);
                out.set(0, j, s);
            }
            return out;
        } else {
            const out = new Matrix(this.rows, 1);
            for (let i = 0; i < this.rows; i++) {
                let s = 0;
                for (let j = 0; j < this.cols; j++) s += this.get(i, j);
                out.set(i, 0, s);
            }
            return out;
        }
    }

    totalSum(): number {
        let s = 0;
        for (let i = 0; i < this.data.length; i++) s += this.data[i];
        return s;
    }

    max(): { value: number; row: number; col: number } {
        let maxVal = -Infinity, row = 0, col = 0;
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
                if (this.get(i, j) > maxVal) {
                    maxVal = this.get(i, j);
                    row = i; col = j;
                }
        return { value: maxVal, row, col };
    }

    slice(rowStart: number, rowEnd: number, colStart: number, colEnd: number): Matrix {
        const out = new Matrix(rowEnd - rowStart, colEnd - colStart);
        for (let i = rowStart; i < rowEnd; i++)
            for (let j = colStart; j < colEnd; j++)
                out.set(i - rowStart, j - colStart, this.get(i, j));
        return out;
    }

    // ---------------- DECOMPOSITIONS ----------------
    static cholesky(A: Matrix): Matrix {
        if (A.rows !== A.cols) throw new Error("Matrix must be square");
        if (!A.isSymmetric()) throw new Error("Matrix is not symmetric");
        const N = A.rows;
        const L = Matrix.identity(N);
        for (let n = 0; n < N; n++) {
            let sum2 = 0;
            for (let m = 0; m <= n; m++) {
                if (n === m) {
                    let s = A.get(n, n) - sum2;
                    if (s < 0) throw new Error("Matrix is not positive definite");
                    L.set(n, n, Math.sqrt(s));
                } else {
                    let sum = 0;
                    for (let k = 0; k < m; k++) sum += L.get(n, k) * L.get(m, k);
                    L.set(n, m, (A.get(n, m) - sum) / L.get(m, m));
                    sum2 += L.get(n, m) * L.get(n, m);
                }
            }
        }
        return L;
    }

    static lu(A: Matrix): { L: Matrix; U: Matrix } {
        const N = A.rows;
        if (N !== A.cols) throw new Error("Matrix must be square");
        let A_old = A.clone(), L = Matrix.identity(N);
        for (let n = 0; n < N - 1; n++) {
            const Mn = Matrix.identity(N);
            const Mn_inv = Matrix.identity(N);
            for (let i = n + 1; i < N; i++) {
                const val = A_old.get(i, n) / A_old.get(n, n);
                Mn.set(i, n, -val);
                Mn_inv.set(i, n, val);
            }
            A_old = Mn.multiply(A_old);
            L = L.multiply(Mn_inv);
        }
        return { L, U: A_old };
    }

    lup(): { L: Matrix; U: Matrix; P: number[] } {
        const n = this.rows;
        if (n !== this.cols) throw new Error("Square only");
        const A = this.clone();
        const P = Array.from({ length: n }, (_, i) => i);
        for (let i = 0; i < n; i++) {
            let max = i, maxVal = Math.abs(A.get(i, i));
            for (let k = i + 1; k < n; k++) {
                const val = Math.abs(A.get(k, i));
                if (val > maxVal) { maxVal = val; max = k; }
            }
            if (Matrix.isZero(maxVal)) throw new Error("Singular matrix");
            if (max !== i) {
                for (let j = 0; j < n; j++) {
                    const tmp = A.get(i, j);
                    A.set(i, j, A.get(max, j));
                    A.set(max, j, tmp);
                }
                [P[i], P[max]] = [P[max], P[i]];
            }
            for (let j = i + 1; j < n; j++) {
                const factor = A.get(j, i) / A.get(i, i);
                A.set(j, i, factor);
                for (let k = i + 1; k < n; k++) A.set(j, k, A.get(j, k) - factor * A.get(i, k));
            }
        }
        const L = Matrix.identity(n), U = Matrix.zeros(n, n);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                if (i > j) L.set(i, j, A.get(i, j));
                else U.set(i, j, A.get(i, j));
        return { L, U, P };
    }

    static luPivotingTotal(A: Matrix): { P: Matrix; L: Matrix; U: Matrix } {
        const n = A.rows;
        if (n !== A.cols) throw new Error("Matrix must be square");
        let A_old = A.clone(), M = Matrix.identity(n), P = Matrix.identity(n);
        for (let col = 0; col < n - 1; col++) {
            let maxVal = 0, pivotRow = col;
            for (let i = col; i < n; i++) {
                const val = Math.abs(A_old.get(i, col));
                if (val > maxVal) { maxVal = val; pivotRow = i; }
            }
            if (Matrix.isZero(maxVal)) continue;
            if (pivotRow !== col) {
                const Pn = Matrix.identity(n);
                Pn.set(col, col, 0); Pn.set(pivotRow, pivotRow, 0);
                Pn.set(col, pivotRow, 1); Pn.set(pivotRow, col, 1);
                P = Pn.multiply(P); A_old = Pn.multiply(A_old);
            }
            const Ltilden = Matrix.identity(n);
            for (let i = col + 1; i < n; i++) Ltilden.set(i, col, -A_old.get(i, col) / A_old.get(col, col));
            M = Ltilden.multiply(M); A_old = Ltilden.multiply(A_old);
        }
        const U = A_old;
        const L = P.multiply(M.inverse());
        return { P, L, U };
    }

    // ---------------- SOSTITUZIONI ----------------
    static solveLowerTriangular(L: Matrix, b: Matrix): Matrix {
        const N = L.rows;
        if (N !== L.cols) throw new Error("Matrix L must be square");
        if (L.subtract(L.tril()).totalSum() > 1e-15) throw new Error("Matrix L is not lower triangular");
        const x = Matrix.zeros(N, b.cols);
        for (let j = 0; j < b.cols; j++) {
            x.set(0, j, b.get(0, j) / L.get(0, 0));
            for (let i = 1; i < N; i++) {
                let sum = 0;
                for (let k = 0; k < i; k++) sum += L.get(i, k) * x.get(k, j);
                x.set(i, j, (b.get(i, j) - sum) / L.get(i, i));
            }
        }
        return x;
    }

    static solveUpperTriangular(U: Matrix, b: Matrix): Matrix {
        const N = U.rows;
        if (N !== U.cols) throw new Error("Matrix U must be square");
        if (U.subtract(U.triu()).totalSum() > 1e-15) throw new Error("Matrix U is not upper triangular");
        const x = Matrix.zeros(N, b.cols);
        for (let j = 0; j < b.cols; j++) {
            x.set(N - 1, j, b.get(N - 1, j) / U.get(N - 1, N - 1));
            for (let i = N - 2; i >= 0; i--) {
                let sum = 0;
                for (let k = i + 1; k < N; k++) sum += U.get(i, k) * x.get(k, j);
                x.set(i, j, (b.get(i, j) - sum) / U.get(i, i));
            }
        }
        return x;
    }

    // ---------------- RISOLUZIONE SISTEMI ----------------
    solve(b: Matrix): Matrix {
        const { L, U, P } = this.lup();
        const n = this.rows;
        const Pb = new Matrix(n, b.cols);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < b.cols; j++)
                Pb.set(i, j, b.get(P[i], j));
        const y = Matrix.solveLowerTriangular(L, Pb);
        return Matrix.solveUpperTriangular(U, y);
    }

    // ---------------- DETERMINANTE E INVERSE ----------------
    det(): number {
        const { U, P } = this.lup();
        let det = 1, sign = 1;
        for (let i = 0; i < P.length; i++) if (P[i] !== i) sign *= -1;
        for (let i = 0; i < this.rows; i++) det *= U.get(i, i);
        return det * sign;
    }

    inverse(): Matrix {
        return this.solve(Matrix.identity(this.rows));
    }

    // ---------------- STAMPA ----------------
    toString(): string {
        let s = "";
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) s += this.get(i, j).toFixed(3) + " ";
            s += "\n";
        }
        return s;
    }
}