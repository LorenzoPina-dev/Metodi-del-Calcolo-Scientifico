# Changelog

Tutte le modifiche degne di nota vengono documentate in questo file.
Il formato segue [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Il versionamento segue [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2025

### Aggiunto

- `Matrix<T>` generico su `INumeric<T>`: Float64M, Complex, Rational.
- Decomposizioni: LUP (pivoting parziale), LU, QR (Gram-Schmidt modificato), Cholesky.
- Solver: LUP, LU, QR, Cholesky, Jacobi (element-wise), Gauss-Seidel (element-wise).
- `ct()` — trasposta coniugata (Hermitian adjoint) corretta per matrici complesse.
- `conjugate()` nell'interfaccia `INumeric<T>` (auto-coniugato per Float64M e Rational).
- Fast-path Float64Array in `mul`, `add`, `sub`, `norm`, `statistics`.
- GCD binario di Lehmer e cross-cancellation in `Rational.add/multiply`.
- Memoization lazy di `magnitudeSq` e `magnitude` in `Complex`.
- Gallery: 23 matrici note (Hilbert, Pascal, Magic, Cauchy, Frank, Grcar, Kahan, Wathen…).
- Classificazione struttura in una passata O(n²) in `det` e `smartInverse`.
- `likeWithData()` per costruzione output senza doppia allocazione.
- Dual-output build: ESM (`dist/esm/`) + CJS (`dist/cjs/`) + tipi (`dist/types/`).
