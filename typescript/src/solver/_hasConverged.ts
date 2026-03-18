import { Matrix } from "../core";

/**
 * Helper privato per verificare la convergenza, aggiornato con il conteggio da 1.
 */
export function _hasConverged(oldX: Matrix, newX: Matrix, tol: number): boolean {
    let maxDiff = 0;
    for (let i = 1; i <= oldX.rows; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(newX.get(i - 1, 0) - oldX.get(i - 1, 0)));
    }
    return maxDiff < tol;
}