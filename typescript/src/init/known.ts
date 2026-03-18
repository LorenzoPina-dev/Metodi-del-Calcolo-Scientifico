import { Matrix } from "../core";
import { qr } from "../decomposition";
import { identity, zeros } from "./init";
import { random } from "./random";

export function hilbert(n: number): Matrix {
    const H = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            H.set(i, j, 1 / (i + j + 1));
    return H;
}

export function pascal(n: number): Matrix {
    if(n > 30)  console.warn("[PRECISION_WARNING] n is large, may cause overflow in binomial coefficients");
    const A = zeros(n, n);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A.set(i, j, nchoosek(i + j, i));
        }
    }

    return A;
}
function nchoosek(n: number, k: number): number {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;

    // usa simmetria per ridurre i passaggi
    if (k > n - k) k = n - k;

    let result = 1;
    for (let i = 1; i <= k; i++) {
        result *= (n - i + 1);
        result /= i;
    }

    return result;
}


export function magic(n: number): Matrix {
    if (n < 3) throw new Error("n must be >= 3");

    if (n % 2 === 1) return magicOdd(n);
    if (n % 4 === 0) return magicDoublyEven(n);

    return magicSinglyEven(n);
}
function magicOdd(n: number): Matrix {
    const M = zeros(n, n);

    let i = 0;
    let j = Math.floor(n / 2);

    for (let k = 1; k <= n * n; k++) {
        M.set(i, j, k);

        const ni = (i - 1 + n) % n;
        const nj = (j + 1) % n;

        if (M.get(ni, nj) !== 0) {
            i = (i + 1) % n;
        } else {
            i = ni;
            j = nj;
        }
    }

    return M;
}
function magicDoublyEven(n: number): Matrix {
    const M = zeros(n, n);

    let num = 1;
    let max = n * n;

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (
                (i % 4 === j % 4) ||
                ((i % 4 + j % 4) === 3)
            ) {
                M.set(i, j, max - num + 1);
            } else {
                M.set(i, j, num);
            }
            num++;
        }
    }

    return M;
}
function magicSinglyEven(n: number): Matrix {
    const p = n / 2;
    const k = (n - 2) / 4;

    const A = magicOdd(p);

    const M = zeros(n, n);

    // blocchi
    for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
            const a = A.get(i, j);

            M.set(i, j, a);
            M.set(i + p, j, a + 3 * p * p);
            M.set(i, j + p, a + 2 * p * p);
            M.set(i + p, j + p, a + p * p);
        }
    }

    // swap colonne
    for (let i = 0; i < p; i++) {
        for (let j = 0; j < k; j++) {
            swap(M, i, j, i + p, j);
        }

        for (let j = n - k + 1; j < n; j++) {
            swap(M, i, j, i + p, j);
        }
    }

    // swap centrale
    swap(M, k, 0, k + p, 0);
    swap(M, k, k, k + p, k);

    return M;
}
function swap(M: Matrix, i1: number, j1: number, i2: number, j2: number) {
    const tmp = M.get(i1, j1);
    M.set(i1, j1, M.get(i2, j2));
    M.set(i2, j2, tmp);
}

export function cauchy(x: number[], y: number[]): Matrix {
    const n = x.length;
    const m = y.length;
    const C = zeros(n, m);

    for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
            const denom = x[i] + y[j];
            if (Math.abs(denom) < Matrix.EPS)
                throw new Error("Division by zero in Cauchy matrix");
            C.set(i, j, 1 / denom);
        }

    return C;
}
export function circul(v: number[]): Matrix {
    const n = v.length;
    const C = zeros(n, n);

    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            C.set(i, j, v[(j - i + n) % n]);

    return C;
}
export function lehmer(n: number): Matrix {
    const A = zeros(n, n);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const num = Math.min(i + 1, j + 1);
            const den = Math.max(i + 1, j + 1);
            A.set(i, j, num / den);
        }
    }

    return A;
}

export function grcar(n: number, k: number = 3): Matrix {
    const G = zeros(n, n);

    for (let i = 0; i < n; i++) {
        G.set(i, i, 1);

        for (let j = 1; j <= k; j++) {
            if (i + j < n) G.set(i, i + j, 1);
        }

        if (i > 0) G.set(i, i - 1, -1);
    }

    return G;
}
export function wilk(n: number): Matrix {
    const W = zeros(n, n);
    const mid = (n - 1) / 2;

    for (let i = 0; i < n; i++) {
        W.set(i, i, Math.abs(i - mid));

        if (i > 0) W.set(i, i - 1, 1);
        if (i < n - 1) W.set(i, i + 1, 1);
    }

    return W;
}
export function house(x: number[]): Matrix {
    const n = x.length;
    const v = [...x];

    let norm = Math.sqrt(v.reduce((s, xi) => s + xi * xi, 0));
    if (norm === 0) throw new Error("Zero vector");

    v[0] += Math.sign(v[0]) * norm;

    const beta = v.reduce((s, xi) => s + xi * xi, 0);

    const H = identity(n);

    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            H.set(i, j, H.get(i, j) - (2 / beta) * v[i] * v[j]);

    return H;
}
//SONO COMPLESSI, VA RIVISTO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
export function smoke(n: number): Matrix {
    const H = zeros(n, n); 

    for (let i = 1; i <= n; i++) {
        // Indici per l'accesso (regolati per array 0-indexed di JS)
        const row = i - 1;
        const col = i - 1;

        // 2. Imposta la diagonale principale: cos(2*PI * i / n)
        const diagValue = Math.cos((2 * Math.PI * i) / n);
        H.set(row, col, diagValue);

        // 3. Imposta la sovradiagonale a 1 (se non siamo all'ultima riga)
        if (i < n) {
            H.set(row, col + 1, 1.0);
        }
    }
    
    return H;
}
export function orthog(n: number): Matrix {
    // Gram-Schmidt su matrice random
    const A = random(n, n);

    const Q = zeros(n, n);

    for (let j = 0; j < n; j++) {
        let v = Array.from({ length: n }, (_, i) => A.get(i, j));

        for (let k = 0; k < j; k++) {
            let dot = 0;
            for (let i = 0; i < n; i++) dot += v[i] * Q.get(i, k);

            for (let i = 0; i < n; i++) v[i] -= dot * Q.get(i, k);
        }

        let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        for (let i = 0; i < n; i++) Q.set(i, j, v[i] / norm);
    }

    return Q;
}

    /**
 * Genera una matrice di Wathen nx per ny.
 * Dimensione finale N = 3*nx*ny + 2*nx + 2*ny + 1.
 */
export function wathen(nx: number, ny: number): Matrix {

    const n = 3 * nx * ny + 2 * nx + 2 * ny + 1;
    const A = zeros(n, n);

    const em = [
        [ 6, -6,  2, -8,  3, -8,  2, -6 ],
        [ -6, 32, -6, 20, -8, 16, -8, 20 ],
        [ 2, -6,  6, -6,  2, -8,  3, -8 ],
        [ -8, 20, -6, 32, -6, 20, -8, 16 ],
        [ 3, -8,  2, -6,  6, -6,  2, -8 ],
        [ -8, 16, -8, 20, -6, 32, -6, 20 ],
        [ 2, -8,  3, -8,  2, -6,  6, -6 ],
        [ -6, 20, -8, 16, -8, 20, -6, 32 ]
    ];

    const node = new Array<number>(8);

    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {

            // ⚠️ Traduzione 1:1 (attenzione agli indici)
            node[0] = (3 * (j + 1)) * nx + 2 * (j + 1) + 2 * (i + 1) + 1 - 1;
            node[1] = node[0] - 1;
            node[2] = node[0] - 2;

            node[3] = (3 * (j + 1) - 1) * nx + 2 * (j + 1) + (i + 1) - 1 - 1;
            node[7] = node[3] + 1;

            node[4] = (3 * (j + 1) - 3) * nx + 2 * (j + 1) + 2 * (i + 1) - 3 - 1;
            node[5] = node[4] + 1;
            node[6] = node[4] + 2;

            const rho = 100 * Math.random();

            for (let krow = 0; krow < 8; krow++) {
                for (let kcol = 0; kcol < 8; kcol++) {
                    const r = node[krow];
                    const c = node[kcol];
                    A.set(r, c, A.get(r, c) + rho * em[krow][kcol]);
                }
            }
        }
    }

    return A;
}


/* ---------- GENERATORI MATRICI ---------- */


export function fiedler(c: number[] | number): Matrix {
    let vec: number[];
    if (typeof c === "number") {
        vec = Array.from({ length: c }, (_, i) => i + 1); // 1:c
    } else {
        vec = c;
    }

    const n = vec.length;
    const A = zeros(n, n);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A.set(i, j, Math.abs(vec[j] - vec[i]));
        }
    }

    return A;
}

/**
 * Tridiagonal matrix: A with sub, main, super diagonals
 */
export function tridiag(a: number[], b: number[], c: number[]): Matrix {
    const n = b.length;
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        A.set(i, i, b[i]);
        if (i > 0) A.set(i, i - 1, a[i - 1]);
        if (i < n - 1) A.set(i, i + 1, c[i]);
    }
    return A;
}

interface Complex {
    re: number;
    im: number;
}

function generateSmokeMatrix(n: number): Complex[][] {
    // Inizializziamo la matrice con zeri complessi
    const matrix: Complex[][] = Array.from({ length: n }, () => 
        Array.from({ length: n }, () => ({ re: 0, im: 0 }))
    );

    for (let j = 1; j <= n; j++) {
        // 1. Diagonale principale: radici dell'unità w^(j-1)
        // Formula: e^(i * theta) = cos(theta) + i * sin(theta)
        const theta = (2 * Math.PI * (j - 1)) / n;
        matrix[j - 1][j - 1] = {
            re: Math.cos(theta),
            im: Math.sin(theta)
        };

        // 2. Sopra-diagonale: 1
        if (j < n) {
            matrix[j - 1][j] = { re: 1, im: 0 };
        }
    }

    // 3. Elemento d'angolo (ultima riga, prima colonna)
    if (n > 1) {
        matrix[n - 1][0] = { re: 1, im: 0 };
    }

    return matrix;
}
/**
 * Generates the Dorr matrix n-by-n, MATLAB equivalent gallery('dorr', n, theta)
 * @param n size
 * @param theta default 0.01
 */
export function dorr(n: number, theta = 0.01): Matrix {
    const h = 1 / (n + 1);
    const m = Math.floor((n + 1) / 2);
    const term = theta / (h * h);

    const c = new Array(n).fill(0); // subdiagonal
    const e = new Array(n).fill(0); // superdiagonal
    const d = new Array(n).fill(0); // main diagonal

    // Primo blocco i = 1:m
    for (let i = 0; i < m; i++) {
        c[i] = -term;
        e[i] = c[i] - (0.5 - (i + 1) * h) / h;
        d[i] = -(c[i] + e[i]);
    }

    // Secondo blocco i = m+1:n
    for (let i = m; i < n; i++) {
        e[i] = -term;
        c[i] = e[i] + (0.5 - (i + 1) * h) / h;
        d[i] = -(c[i] + e[i]);
    }

    // Ridimensionamento dei vettori per la tridiagonale
    const sub = c.slice(1, n);      // subdiagonale n-1
    const sup = e.slice(0, n - 1);  // superdiagonale n-1

    // Costruzione matrice tridiagonale
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        A.set(i, i, d[i]);
        if (i < n - 1) {
            A.set(i, i + 1, sup[i]);
            A.set(i + 1, i, sub[i]);
        }
    }

    return A;
}


/**
 * Neumann matrix: discretizzazione Laplaciana su griglia quadrata
 */
export function neumann(n: number): Matrix {
    const k = Math.sqrt(n);
    
    if (!Number.isInteger(k)) {
        throw new Error("Il parametro n deve essere un quadrato perfetto (es. 4, 9, 16).");
    }

    const matrix: Matrix = zeros(n, n); 

    for (let p = 1; p <= n; p++) {
        // Calcoliamo le coordinate (i, j) nella griglia k x k (1-based)
        const i = Math.ceil(p / k);
        const j = ((p - 1) % k) + 1;

        // 1. Diagonale principale: sempre 4
        matrix.set(p - 1, p - 1, 4);

        // 2. Vicini Orizzontali (Sinistra/Destra)
        if (k > 1) {
            if (j === 1) {
                // Bordo sinistro: riflette sul vicino a destra (p+1)
                matrix.set(p - 1, (p + 1) - 1, matrix.get(p - 1, (p + 1) - 1) - 2);
            } else if (j === k) {
                // Bordo destro: riflette sul vicino a sinistra (p-1)
                matrix.set(p - 1, (p - 1) - 1, matrix.get(p - 1, (p - 1) - 1) - 2);
            } else {
                // Nodo interno orizzontalmente: entrambi i vicini pesano -1
                matrix.set(p - 1, (p - 1) - 1, matrix.get(p - 1, (p - 1) - 1) - 1);
                matrix.set(p - 1, (p + 1) - 1, matrix.get(p - 1, (p + 1) - 1) - 1);
            }
        }

        // 3. Vicini Verticali (Sopra/Sotto)
        if (k > 1) {
            if (i === 1) {
                // Bordo superiore: riflette sul vicino sotto (p+k)
                matrix.set(p - 1, (p + k) - 1, matrix.get(p - 1, (p + k) - 1) - 2);
            } else if (i === k) {
                // Bordo inferiore: riflette sul vicino sopra (p-k)
                matrix.set(p - 1, (p - k) - 1, matrix.get(p - 1, (p - k) - 1) - 2);
            } else {
                // Nodo interno verticalmente: entrambi i vicini pesano -1
                matrix.set(p - 1, (p - k) - 1, matrix.get(p - 1, (p - k) - 1) - 1);
                matrix.set(p - 1, (p + k) - 1, matrix.get(p - 1, (p + k) - 1) - 1);
            }
        }
    }

    return matrix;
}

export function invhess(n: number): Matrix {
    const A = zeros(n, n);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (j < i) {
                A.set(i, j, j+1);      // sotto diagonale
            } else if (i === j) {
                A.set(i, j, i + 1);  // diagonale principale
            } else {
                A.set(i, j, -(i + 1)); // sopra diagonale
            }
        }
    }

    return A;
}
/**
 * Genera la matrice gallery('binomial', n, p)
 * @param n Dimensione della matrice
 * @param p Parametro scalare
 */
export function binomial(n: number, p: number): Matrix {
    const matrix: Matrix = zeros(n, n);

    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= n; j++) {
            let sum = 0;
            // Sommiamo sui possibili indici k
            for (let k = 1; k <= n; k++) {
                const term1 = combinations(i - 1, k - 1);
                const term2 = combinations(n - i, j - k);
                const sign = (k-1) % 2 === 0 ? 1 : -1;
                
                sum += term1 * term2 * sign;
            }
            matrix.set(i - 1, j - 1,  sum);
        }
    }

    return matrix;
}

/**
 * Funzione di supporto per il coefficiente binomiale (n su k)
 */
function combinations(n: number, k: number): number {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    if (k > n / 2) k = n - k;
    let res = 1;
    for (let i = 1; i <= k; i++) {
        res = res * (n - i + 1) / i;
    }
    return Math.round(res);
}
/**
 * Genera una matrice n x m con distribuzione normale standard (media 0, varianza 1)
 */
function generateNormalDistributionMatrix(rows: number, cols: number): Matrix {
    const matrix: Matrix = zeros(rows, cols);

    for (let i = 1; i <= rows; i++) {
        for (let j = 1; j <= cols; j++) {
            matrix.set(i - 1, j - 1, gaussianRandom());
        }
    }

    return matrix;
}

/**
 * Genera un singolo numero casuale con distribuzione normale (Box-Muller)
 */
function gaussianRandom(): number {
    let u = 0, v = 0;
    // Math.random() può restituire 0, ma ln(0) è indefinito, quindi saltiamo lo zero.
    while (u === 0) u = Math.random(); 
    while (v === 0) v = Math.random();
    
    // Applichiamo la trasformazione di Box-Muller
    const z0 = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z0;
}
/**
 * Random SVD matrix: generate random orthogonal U and V, singular values 1..n
 */
export function randsvd(n: number): Matrix {
    let kappa =Math.sqrt(1/Number.EPSILON); // default kappa molto grande per matrice quasi singolare
    // 1. Genera matrici casuali con distribuzione normale
    let G1 = generateNormalDistributionMatrix(n, n);
    let G2 = generateNormalDistributionMatrix(n, n);

    // 2. Ottieni U e V tramite decomposizione QR
    let { Q: U } = qr(G1);
    let { Q: V } = qr(G2);

    // 3. Crea la diagonale Sigma basata su kappa e mode
    let sigma = new Float64Array(n);
    for (let i = 1; i <= n; i++) {
        // Esempio: distribuzione geometrica
        sigma[i - 1] = Math.pow(kappa, -(i - 1) / (n - 1));
    }
    // 4. A = U * diag(sigma) * V^T
    return U.multiply(new Matrix(n, n, sigma).multiply(V.transpose()));
}

// Crea la matrice minij n×n: min(i,j)
function minij(n: number): Matrix {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A.set(i, j, Math.min(i + 1, j + 1)); // +1 perché R usa indici 1-based
        }
    }
    return A;
}

// Estrae la parte triangolare superiore della matrice con offset k
function triu(A: Matrix, k = 0): Matrix {
    const n = A.rows;
    const m = A.cols;
    const B = zeros(n, m);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
            if (j - i >= k) {
                B.set(i, j, A.get(i, j));
            }
        }
    }
    return B;
}

// Funzione Frank
export function frank(n: number, k = 0): Matrix {
    let F = minij(n);
    F = triu(F, -1);

    if (k === 0) {
        const p = Array.from({ length: n }, (_, i) => n - i - 1); // R indice inverso 1:n
        const G = zeros(n, n);
        // Riflettiamo sulla anti-diagonale
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                G.set(i, j, F.get(p[i], p[j]));
            }
        }
        // Trasponiamo
        const H = zeros(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                H.set(i, j, G.get(j, i));
            }
        }
        F = H;
    }

    return F;
}

export function hanowa(n: number, d = -1): Matrix {
    if (n % 2 !== 0) {
        throw new Error("hanowa: n must be even");
    }

    const m = n / 2;
    const A = zeros(n, n);

    // Creiamo le matrici diagonali
    const D = identity(m).multiply(d); // d*eye(m)
    const diag1toM = zeros(m, m);
    for (let i = 0; i < m; i++) {
        diag1toM.set(i, i, i + 1); // diag(1:m)
    }

    // Costruzione dei 4 blocchi
    // Blocchi superiori
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
            A.set(i, j, D.get(i, j));           // D
            A.set(i, j + m, -diag1toM.get(i, j)); // -diag(1:m)
        }
    }

    // Blocchi inferiori
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
            A.set(i + m, j, diag1toM.get(i, j)); // diag(1:m)
            A.set(i + m, j + m, D.get(i, j));    // D
        }
    }

    return A;
}

export function kahan(n: number, m: number = n, alpha: number = 1.2, pert: number = 1e3): Matrix {
    const A = zeros(m, n);
    const s = Math.sin(alpha);
    const c = Math.cos(alpha);
    const eps = Number.EPSILON;

    for (let i = 0; i < m; i++) {
        const si = Math.pow(s, i );
        const csi = -c * si;

        for (let j = 0; j < n; j++) {
            if (j === i) {
                A.set(i, j, si + pert * eps * (Math.min(m, n) - i));
            } else if (i < j) {
                A.set(i, j, csi);
            }
            // sotto diagonale rimane zero
        }
    }

    return A;
}