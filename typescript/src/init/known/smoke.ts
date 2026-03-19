import { Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice "Smoke" (Versione Reale)
 * * Descrizione:
 * Una matrice complessa con radici dell'unità sulla diagonale e 1 sulla sovradiagonale.
 * * Nota Critica:
 * La versione MATLAB gallery('smoke') è complessa. Questa implementazione usa solo la parte reale 
 * cos(2π*i/n). Per una fedeltà totale, servirebbe il supporto ai numeri complessi.
 * * Funzionamento:
 * 1. Diagonale: cos(2*PI * i / n).
 * 2. Sovradiagonale: 1.0.
 * 3. Elemento d'angolo (n, 1): 1.0 per "chiudere" il ciclo.
 */

//SONO COMPLESSI, VA RIVISTO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
export function smoke(n: number): Matrix {
    const H = zeros(n, n); 

    for (let i = 1; i <= n; i++) {
        const diagValue = Math.cos((2 * Math.PI * (i - 1)) / n);
        H.set(i - 1, i - 1, diagValue);

        if (i < n) {
            H.set(i - 1, i, 1.0);
        }
    }
    // Elemento circolante nell'angolo in basso a sinistra
    if (n > 1) H.set(n - 1, 0, 1.0);
    
    return H;
}