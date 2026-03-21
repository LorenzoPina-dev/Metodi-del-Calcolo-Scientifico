        it("Q^H * Q = I (colonne ortonormali, usando ct())", () => {
            // Per matrici complesse: Q^H = conj(Q)^T = Q.ct()
            // Q^H * Q deve essere la matrice identità n×n
            const A = cmat([[[1, 1], [2, 0]], [[0, 1], [1, -1]], [[-1, 0], [1, 1]]]);
            const { Q } = Matrix.decomp.qr(A);
            const QhQ = Q.ct().mul(Q);
            const I = Matrix.identityOf(2, Complex.zero, Complex.one);
            expect(mEq(QhQ, I, 1e-9)).toBe(true);
        });
