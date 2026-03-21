import { Float64M, Matrix } from "../..";
import { qr } from "../../decomposition";
import { zeros } from "../init";

/**
 * Matrice con Valori Singolari Casuali
 * * Descrizione:
 * Crea una matrice con un numero di condizionamento (kappa) specificato.
 * * Proprietà:
 * - Permette di testare algoritmi di risoluzione lineare con matrici "difficili".
 * * Funzionamento:
 * 1. Genera due matrici ortogonali U e V tramite QR di matrici gaussiane.
 * 2. Crea una diagonale Sigma con distribuzione geometrica (mode 3).
 * 3. Restituisce A = U * Sigma * V^T.
 */
export function randsvd(n: number): Matrix {
    let kappa = Math.sqrt(1 / Number.EPSILON);
    let G1 = generateNormalDistributionMatrix(n, n);
    let G2 = generateNormalDistributionMatrix(n, n);

    let { Q: U } = qr(G1);
    let { Q: V } = qr(G2);

    let sigma = new Array<Float64M>(n);
    for (let i = 0; i < n; i++) {
        sigma[i] =new Float64M( Math.pow(kappa, -i / (n - 1)));
    }
    // A = U * diag(sigma) * V^T
    const SigmaMat = Matrix.diagFromArray(sigma); // Assumendo costruttore per diagonale
    return U.mul(SigmaMat.mul(V.t()));
}

// Supporto: Box-Muller per distribuzione normale
function gaussianRandom(): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random(); 
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function generateNormalDistributionMatrix(rows: number, cols: number): Matrix {
    const matrix = zeros(rows, cols);
    for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
            matrix.set(i, j, gaussianRandom());
    return matrix;
}