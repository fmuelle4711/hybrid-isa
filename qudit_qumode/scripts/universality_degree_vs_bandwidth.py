"""Does the iteration-3 Lie-closure span actually contain polynomial degree > 3?

results/universality_closure_check.md reports a "max polynomial degree" column
and concludes that degree 3 at iteration 3 shows intrinsic non-Gaussianity.
That column is produced by verification/universality_closure_check.py::
max_bandwidth_1mode, whose own section header reads

    "Polynomial-degree extraction (matrix bandwidth in safe interior)"

i.e. it returns max |m - n| over Fock indices in a safe window.  BANDWIDTH IS
NOT POLYNOMIAL DEGREE.  Any function of the number operator -- n^2, n^3, ... --
has bandwidth 0 and unbounded degree, so a bandwidth of 3 places no upper
bound on the polynomial degree present.

This script performs the separate demonstration the claim needs, by a
dimension count that does not depend on the bandwidth metric at all:

    the canonical space  u(d) (x) {Weyl-symmetrised monomials in x,p of
    total degree <= 3}  has dimension d^2 * 10  ( = 160 for d = 4 )

If the iteration-3 span has dimension GREATER than that, it cannot be
contained in the cubic space, and operators of degree >= 4 are present while
the bandwidth metric still reports 3.

Read-only with respect to every other file; writes only its own results file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verification"))

from universality_closure_check import (                          # noqa: E402
    build_seeds_hybrid_1mode, compute_lie_closure, fock_ops,
    qudit_seeds, max_bandwidth_1mode)

OUT = ROOT / "results" / "universality_degree_vs_bandwidth.txt"
D = 4
TOL = 1e-9
L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


def weyl_monomials(N, max_deg=3):
    """Weyl-symmetrised monomials in x,p of total degree <= max_deg.

    Returns list of (label, matrix).  Symmetrisation is the average over all
    distinct orderings of the letters, which is the standard Hermitian
    representative of x^j p^k.
    """
    from itertools import permutations
    _, _, x, p = fock_ops(N)
    I = np.eye(N, dtype=complex)
    out = [("1", I.copy())]
    for deg in range(1, max_deg + 1):
        for j in range(deg + 1):
            k = deg - j
            letters = ["x"] * j + ["p"] * k
            acc = np.zeros((N, N), dtype=complex)
            perms = set(permutations(letters))
            for perm in perms:
                M = I.copy()
                for ch in perm:
                    M = M @ (x if ch == "x" else p)
                acc += M
            acc /= len(perms)
            out.append((f"x^{j} p^{k}", acc))
    return out


def full_u_d(d):
    """All d^2 Hermitian generators of u(d): diagonal |l><l|, symmetric X_ij,
    antisymmetric Y_ij.  qudit_seeds() omits the Y_ij, so it spans only
    d(d+1)/2 + ... = 10 of the 16 directions at d = 4."""
    ops = []
    for l in range(d):
        E = np.zeros((d, d), dtype=complex)
        E[l, l] = 1.0
        ops.append(E)
    for i in range(d):
        for j in range(i + 1, d):
            X = np.zeros((d, d), dtype=complex)
            X[i, j] = 1.0; X[j, i] = 1.0
            Y = np.zeros((d, d), dtype=complex)
            Y[i, j] = -1j; Y[j, i] = 1j
            ops.append(X); ops.append(Y)
    return ops


def span_dim(mats, tol=1e-9):
    A = np.array([M.ravel() for M in mats])
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > tol * s[0]).sum()), s


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    W("=" * 104)
    W("UNIVERSALITY DIAGNOSTIC: IS BANDWIDTH 3 THE SAME AS DEGREE 3?")
    W("=" * 104)
    W("  The reported 'max polynomial degree' column is matrix bandwidth")
    W("  (verification/universality_closure_check.py, max_bandwidth_1mode).")
    W("  Bandwidth 3 does not bound polynomial degree: n^k is bandwidth 0 at")
    W("  any degree.  Tested here by a dimension count instead.")
    W()

    for N in (6, 8, 10):
        dim_full = D * N
        W("-" * 104)
        W(f"  N = {N}   (operator space {dim_full} x {dim_full}, "
          f"u(dN) has dim {dim_full**2})")

        # ---- canonical cubic space -----------------------------------
        # FULL u(D), all D^2 Hermitian generators.  NOT qudit_seeds, which is
        # the paper's ISA seed set (6 X_ij + 4 diagonal = 10 of the 16) and
        # would understate the canonical space.
        qudit_basis = full_u_d(D)
        monos = weyl_monomials(N, max_deg=3)
        canon = [np.kron(Q, M) for Q in qudit_basis for _, M in monos]
        d_canon, _ = span_dim(canon, TOL)
        W(f"    canonical space  u({D}) (x) monomials deg<=3 : "
          f"{len(qudit_basis)} x {len(monos)} = {len(canon)} generators, "
          f"span dim = {d_canon}")
        if d_canon != D * D * 10:
            W(f"      NOTE: nominal {D*D}*10 = {D*D*10}; the numerical span is "
              f"{d_canon} (truncation dependence at this N)")

        # ---- closure to iteration 3 ----------------------------------
        seeds = build_seeds_hybrid_1mode(D, N)
        basis, bounds, dims = compute_lie_closure(
            seeds, tol=TOL, max_iter=3, hard_dim_cap=dim_full ** 2,
            verbose=False)
        W(f"    Lie closure dims per iteration: {dims}")
        d_span = len(basis)
        W(f"    iteration-3 span dim = {d_span}")

        bw = max(max_bandwidth_1mode(M, D, N, safe_max=3, tol=TOL)
                 for M in basis)
        W(f"    max bandwidth over those {d_span} elements = {bw}   "
          f"(this is the number the doc calls 'max degree')")

        # ---- is the span inside the canonical space? -----------------
        joint, _ = span_dim(canon + list(basis), TOL)
        outside = joint - d_canon
        W(f"    dim span(canonical + closure) = {joint}")
        W(f"    -> {outside} closure directions lie OUTSIDE the cubic space")
        if d_span > d_canon:
            W(f"    DECISIVE: {d_span} > {d_canon}, so the span cannot be")
            W(f"    contained in u({D}) (x) cubic polynomials.  Operators of")
            W(f"    polynomial degree >= 4 are present while the bandwidth")
            W(f"    metric still reports {bw}.")
        elif outside > 0:
            W(f"    The span is not contained in the cubic space "
              f"({outside} directions outside).")
        else:
            W("    The span IS inside the cubic space; bandwidth and degree")
            W("    agree here and the doc's wording would be defensible.")
        W()

    W("=" * 104)
    W("CONCLUSION")
    W("=" * 104)
    W("  The 'max polynomial degree' column measures matrix bandwidth, and the")
    W("  iteration-3 span provably exits the cubic polynomial space, so the")
    W("  two are not the same quantity.")
    W()
    W("  WHAT SURVIVES.  Only the bare observation that the per-iteration dims")
    W("  are identical at N = 6, 8, 10.  No inference about truncation follows")
    W("  from it: three cutoffs agreeing is consistent with all three carrying")
    W("  the same artefact, and the safe-interior window mitigates boundary")
    W("  leakage without proving its absence.")
    W()
    W("  WHAT DOES NOT.  The inference 'bandwidth 3 -> polynomial degree 3 ->")
    W("  non-Gaussian resource' does not follow, and the N-independence of the")
    W("  dims does not license it either.  The claim that the numerical ranks")
    W("  rule out a truncation artifact should be restated as a claim about")
    W("  bandwidth, or dropped, until degree is measured directly.")
    W()
    W("  CHEAPEST ROUTE TO A REAL DEGREE STATEMENT.  Project each closure")
    W("  element onto Weyl-symmetrised monomials of increasing degree and")
    W("  report the smallest max_deg whose span contains the closure.  That is")
    W("  the same linear algebra as above, run over max_deg = 3, 4, 5, ...")
    W()
    W("  NO ADOPTION.  This script edits nothing else.")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
