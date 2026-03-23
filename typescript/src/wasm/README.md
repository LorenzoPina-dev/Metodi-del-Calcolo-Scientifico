# WebAssembly Integration — numeric-matrix

## Setup rapido

```bash
# 1. Compila il modulo WASM (con SIMD)
npm run build:wasm

# 2. Se il motore non supporta SIMD (raro):
npm run build:wasm:nosimd

# 3. Lancia i benchmark
npm run benchmark
npm run benchmark:simple
```

## Architettura

```
src/wasm/
  matrix_ops.ts     ← sorgente AssemblyScript (compilato → .wasm)
  matrix_ops.wasm   ← binario generato (non nel repo, genera con build:wasm)
  wasm_bridge.ts    ← bridge TypeScript ↔ WASM (singleton, bump allocator)
  index.ts          ← re-export pubblico
```

## Gerarchia di performance (Float64M)

Ogni operazione usa questa priorità in automatico:

```
1. WASM + SIMD  ← attivo dopo initWasm()   ~3-8× più veloce
2. TS Float64 fast-path  ← Float64Array, zero allocazioni intermedie
3. Path generico  ← Complex, Rational (correttezza > performance)
```

## Operazioni coperte da WASM

| Categoria          | Operazioni                                                         |
|--------------------|--------------------------------------------------------------------|
| Element-wise       | add, sub, dotMul, dotDiv, addScalar, subScalar, mulScalar, dotPow |
| Broadcast          | addRowVec, subRowVec, addColVec, subColVec, dotMulRowVec/ColVec   |
| Matriciale         | matmul (blocchi 64×64 + SIMD), matvec                             |
| Trasformazioni     | transpose (cache-blocked 32×32)                                    |
| Unarie (SIMD)      | abs, neg, sqrt, floor, ceil — sin/cos/exp/tan (scalare)           |
| Statistiche        | totalSum, trace, sumCols, sumRows, maxCols, minCols, maxRows, minRows |
| Norme              | normFro, normVec1, normVecInf, normMat1, normMatInf               |
| Property checks    | isSymmetric, isUpperTri, isLowerTri, isDiagonal, isZero, hasFinite, isDiagDom |
| Solver triangolari | solveLower, solveLowerUnit, solveUpper                            |
| Decomposizioni     | LUP, Cholesky, QR (MGS), LDLT                                     |
| Solver iterativi   | Jacobi, Gauss-Seidel, SOR, CG (intero loop in WASM, zero GC)     |

## Uso in applicazione

```typescript
import { initWasm } from "./src/wasm/index.js";
import { Matrix } from "./src/index.js";

// Una sola volta all'avvio
await initWasm();

// Da qui tutte le operazioni Float64M usano WASM in automatico
const A = Matrix.random(500, 500);
const B = Matrix.random(500, 500);
const C = A.mul(B);           // matmul WASM a blocchi + SIMD
const x = A.solve(b, "LUP"); // LUP WASM + solver triangolari WASM
const n = A.norm("Fro");      // norm WASM SIMD
```

## Ottimizzazioni WASM

- **SIMD f64x2**: add/sub/mul/div processano 2 double per ciclo CPU
- **Matmul a blocchi 64×64**: riuso cache L1/L2, elimina cache miss
- **Trasposta cache-blocked 32×32**: accessi sequenziali in lettura e scrittura
- **Solver iterativi completi in WASM**: zero boundary JS per iterazione, zero GC pressure
- **Bump allocator**: `alloc()` è O(1), `reset()` libera tutto in O(1)
- **Matvec SIMD**: dot-product con f64x2 accumulator nel CG interno

## Guadagni attesi

| Operazione       | n=100  | n=500  | n=1000 |
|------------------|--------|--------|--------|
| matmul           | ~3×    | ~5×    | ~6-8×  |
| add/sub/dotMul   | ~2×    | ~3×    | ~4×    |
| LUP              | ~3×    | ~5×    | ~6×    |
| Cholesky         | ~3×    | ~5×    | ~6×    |
| QR               | ~3×    | ~4×    | ~5×    |
| Jacobi/GS/SOR    | ~5×    | ~8×    | ~10×   |
| CG               | ~4×    | ~6×    | ~8×    |
| norm Fro         | ~2×    | ~3×    | ~3×    |
| isSymmetric      | ~2×    | ~3×    | ~3×    |
