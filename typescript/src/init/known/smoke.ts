// init/known/smoke.ts
//
// NOTA: La matrice "Smoke" originale di MATLAB ha autovalori complessi
// (radici n-esime dell'unità sulla diagonale).
// Qui vengono fornite DUE varianti:
//   - smokeReal  → usa solo la parte reale cos(2πk/n)  [Matrix<Float64M>]
//   - smokeComplex → usa i numeri complessi esatti       [Matrix<Complex>]
//
import { Float64M, Complex, Matrix } from "../..";
import { zeros, zerosLike } from "../init";

/**
 * Variante reale (approssimazione): diagonale = cos(2π*i/n), sopra-diagonale = 1.
 */
export function smoke(n: number): Matrix<Float64M> {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        A.setNum(i, i, Math.cos((2 * Math.PI * i) / n));
        if (i < n - 1) A.setNum(i, i + 1, 1);
    }
    if (n > 1) A.setNum(n - 1, 0, 1);   // elemento circolante
    return A;
}

/**
 * Variante complessa esatta: diagonale = e^{2πi·k/n}, sopra-diagonale = 1.
 */
export function smokeComplex(n: number): Matrix<Complex> {
    const z = Complex.zero;
    const o = Complex.one;
    const A = zerosLike<Complex>(n, n, z, o);
    for (let i = 0; i < n; i++) {
        const angle = (2 * Math.PI * i) / n;
        A.set(i, i, new Complex(Math.cos(angle), Math.sin(angle)));
        if (i < n - 1) A.set(i, i + 1, o);
    }
    if (n > 1) A.set(n - 1, 0, o);
    return A;
}
