/**
 * Matrix Generators - Known Matrices Collection
 * * Questa collezione include matrici classiche, matrici di test per algoritmi 
 * di algebra lineare e generatori basati su proprietà matematiche specifiche.
 * * Nota: Tutte le funzioni seguono la logica di input basata su dimensioni n (1-based).
 */

// Matrici Classiche e Strutturate
export { hilbert } from "./hilbert";
export { pascal } from "./pascal";
export { magic } from "./magic";
export { circul } from "./circul";
export { lehmer } from "./lehmer";
export { minij } from "./minij";
export { tridiag } from "./tridiag";

// Matrici di Test e Analisi (Gallery)
export { cauchy } from "./cauchy";
export { frank } from "./frank";
export { grcar } from "./grcar";
export { kahan } from "./kahan";
export { invhess } from "./invhess";
export { binomial } from "./binomial";

// Matrici per PDE e Elementi Finiti
export { wathen } from "./wathen";
export { neumann } from "./neumann";
export { dorr } from "./dorr";

// Matrici di Trasformazione e Ortogonali
export { house } from "./house";
export { orthog } from "./orthog";
export { randsvd } from "./randsvd";

// Matrici con Proprietà Spettrali Specifiche
export { fiedler } from "./fiedler";
export { hanowa } from "./hanowa";
export { smoke } from "./smoke";