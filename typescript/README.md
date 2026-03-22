# numeric-matrix

Libreria TypeScript per algebra lineare generica su tre tipi numerici:

| Tipo | Classe | Caratteristica |
|------|--------|----------------|
| Float 64-bit | `Float64M` | Prestazioni native, fast-path Float64Array |
| Complessi | `Complex` | Aritmetica esatta a+bi, QR hermitiano |
| Razionali esatti | `Rational` | Zero errore di arrotondamento, cross-cancellation BigInt |

Gli algoritmi (LUP, QR, Cholesky, Jacobi, Gauss-Seidel) funzionano **trasparentemente** su tutti e tre i tipi tramite l'interfaccia `INumeric<T>`.

## Installazione

```bash
npm install numeric-matrix
```

## Utilizzo rapido

### Float64 (default)

```typescript
import { Matrix } from 'numeric-matrix';

const A = Matrix.fromArray([[4, 3], [6, 3]]);
const b = Matrix.fromArray([[10], [12]]);

const x = A.solve(b);            // LUP default
console.log(x.get(0, 0).value);  // 1
console.log(x.get(1, 0).value);  // 2

console.log(A.det().value);      // -6
console.log(A.norm('Fro'));      // number
```

### Numeri complessi

```typescript
import { Matrix, Complex } from 'numeric-matrix';

const A = Matrix.zerosOf(2, 2, Complex.zero, Complex.one);
A.set(0, 0, new Complex(2, 1));
A.set(0, 1, new Complex(1, 0));
A.set(1, 0, new Complex(0, 1));
A.set(1, 1, new Complex(1, -1));

const b = Matrix.zerosOf(2, 1, Complex.zero, Complex.one);
b.set(0, 0, new Complex(3, 2));
b.set(1, 0, new Complex(1, 1));

const x = A.solve(b, 'QR');   // usa Q^H (adjoint) correttamente
```

### Aritmetica razionale esatta

```typescript
import { Matrix, Rational } from 'numeric-matrix';

// Matrice di Hilbert 4×4 in aritmetica esatta
const H = Matrix.zerosOf(4, 4, Rational.zero, Rational.one);
for (let i = 0; i < 4; i++)
  for (let j = 0; j < 4; j++)
    H.set(i, j, new Rational(1, i + j + 1));

const b = Matrix.zerosOf(4, 1, Rational.zero, Rational.one);
for (let i = 0; i < 4; i++) b.set(i, 0, Rational.one);

const x = H.solve(b);           // soluzione ESATTA, zero errore floating
const Hx = H.mul(x);

// Hx === b esattamente (confronto bigint, non floating point)
```

## API

### `Matrix<T>` — metodi principali

| Metodo | Descrizione |
|--------|-------------|
| `add(B)` / `sub(B)` / `mul(B)` | Operazioni matriciali + broadcasting vettori |
| `dotMul(B)` / `dotDiv(B)` / `dotPow(e)` | Operazioni element-wise |
| `pow(n)` | Potenza matriciale (binary exponentiation) |
| `det()` | Determinante → tipo `T` |
| `trace()` | Traccia → tipo `T` |
| `norm(type)` | Norma (`'1'`, `'Inf'`, `'Fro'`) → `number` |
| `inv()` | Inversa (smart: diagonale → triangolare → LUP) |
| `solve(b, method?)` | Sistema lineare (`'LUP'`, `'LU'`, `'QR'`, `'Cholesky'`, `'Jacobi'`, `'Gauss-Seidel'`) |
| `t()` | Trasposta |
| `ct()` | Trasposta coniugata / Hermitian adjoint (= `t()` per tipi reali) |
| `reshape(r, c)` | Cambia forma |
| `slice(r0, r1, c0, c1)` | Sotto-matrice |
| `flip(dim)` | Ribalta (flipud / fliplr) |
| `rot90(k)` | Rotazione 90° |
| `abs()` / `sqrt()` / `round()` / `negate()` | Unarie element-wise |
| `exp()` / `sin()` / `cos()` / `tan()` / `floor()` / `ceil()` | Unarie matematiche |
| `sum(dim)` / `mean(dim)` / `max(dim)` / `min(dim)` | Statistiche |
| `isSquare()` / `isSymmetric()` / `isDiagonal()` … | Proprietà strutturali |

### Factory statiche

```typescript
Matrix.zeros(r, c)              // Float64M
Matrix.ones(r, c)               // Float64M
Matrix.identity(n)              // Float64M
Matrix.fromArray([[1,2],[3,4]]) // Float64M da number[][]
Matrix.zerosOf(r, c, zero, one) // tipo generico T
Matrix.identityOf(n, zero, one) // tipo generico T
Matrix.fromTypedArray(data, z, o) // da T[][]
```

### Decomposizioni

```typescript
const { L, U, P, swaps } = Matrix.decomp.lup(A);
const { L, U }            = Matrix.decomp.lu(A);
const { L }               = Matrix.decomp.cholesky(A);  // A SPD
const { Q, R }            = Matrix.decomp.qr(A);
```

### Gallery (matrici note)

```typescript
Matrix.gallery.hilbert(n)
Matrix.gallery.pascal(n)
Matrix.gallery.magic(n)
Matrix.gallery.tridiag(a, b, c)
Matrix.gallery.cauchy(x, y)
// ... e altre 20+
```

### Interfaccia `INumeric<T>`

Per implementare un tipo numerico personalizzato:

```typescript
import type { INumeric } from 'numeric-matrix';

class MyNumber implements INumeric<MyNumber> {
  readonly kind = 'mynumber';
  // ... implementa: add, subtract, multiply, divide, negate,
  //     abs, sqrt, round, conjugate,
  //     greaterThan, lessThan, equals, isNearZero,
  //     toNumber, fromNumber, toString
}
```

## Prestazioni

- **Float64M**: fast-path con `Float64Array` nei loop critici (mul, add, norm). Moltiplicazione 100×100: ~×50 più veloce del path generico.
- **Complex**: `magnitudeSq` e `magnitude` memoizzati; confronti usano il quadrato del modulo (evitano `sqrt`).
- **Rational**: GCD binario di Lehmer; cross-cancellation in `add`/`multiply` mantiene i denominatori piccoli.

## Tipi e moduli

```
numeric-matrix
├── Matrix<T>          — classe principale
├── Float64M           — tipo numerico float64
├── Complex            — tipo numerico complesso
├── Rational           — tipo numerico razionale esatto (BigInt)
└── INumeric<T>        — interfaccia per tipi custom
```

## Licenza

MIT
