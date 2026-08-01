"""The trace algebra is the measurement. Every identity is checked against a
direct dense computation of the quantity it claims to equal.

If one of these fails, the telemetry is not measuring what the docstrings say
it measures, and no amount of downstream analysis will recover it.
"""

from __future__ import annotations

import math

import pytest
import torch

from sftlens.telemetry.reductions import (
    compute_traces,
    derive_metrics,
    gram,
    mp_null_pr,
    shuffle_null,
    top_eigenvalues,
)

TOL = 1e-9


@pytest.fixture
def case():
    """Small, non-square, fp64 so the identities are tested and not the noise."""
    torch.manual_seed(0)
    n, d_in, d_out = 200, 37, 23
    X = torch.randn(n, d_in, dtype=torch.float64)
    Delta = torch.randn(n, d_out, dtype=torch.float64)
    Xc = X - X.mean(0, keepdim=True)
    return {
        "n": n, "d_in": d_in, "d_out": d_out,
        "X": X, "Xc": Xc, "Delta": Delta,
        "K": Xc @ Xc.T, "M": Delta @ Delta.T,
        "Sigma": Xc.T @ Xc / (n - 1),
        "Omega": Delta.T @ Delta,
        "G": Delta.T @ Xc,      # gradient formed from the CENTRED inputs
    }


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def test_raw_traces_match_dense(case):
    K, M = case["K"], case["M"]
    tr = compute_traces(K, M)
    K2, M2 = K @ K, M @ M

    assert _rel(tr["trK"], torch.trace(K).item()) < TOL
    assert _rel(tr["trK2"], torch.trace(K2).item()) < TOL
    assert _rel(tr["trM"], torch.trace(M).item()) < TOL
    assert _rel(tr["trM2"], torch.trace(M2).item()) < TOL
    assert _rel(tr["trKM"], torch.trace(K @ M).item()) < TOL
    assert _rel(tr["trK2M"], torch.trace(K2 @ M).item()) < TOL
    assert _rel(tr["trK3M"], torch.trace(K2 @ K @ M).item()) < TOL
    assert _rel(tr["trK2M2"], torch.trace(K2 @ M2).item()) < TOL


def test_tr_k2m2_is_not_tr_km_squared(case):
    """Guards the specific error the two-matmul reformulation invites.

    tr(K^2 M^2) = ||KM||_F^2, NOT tr((KM)^2). K and M do not commute, so the
    two differ by O(1) and swapping them would corrupt the full K-FAC Rayleigh
    quotient without changing anything else.
    """
    K, M = case["K"], case["M"]
    P = K @ M
    correct = torch.trace(K @ K @ M @ M).item()
    wrong = torch.trace(P @ P).item()

    assert _rel(compute_traces(K, M)["trK2M2"], correct) < TOL
    assert _rel(wrong, correct) > 0.1, "test case fails to separate the two forms"


def test_derived_metrics_match_dense_definitions(case):
    n = case["n"]
    Sigma, Omega, G = case["Sigma"], case["Omega"], case["G"]
    m = derive_metrics(compute_traces(case["K"], case["M"]), n, case["d_in"], case["d_out"])

    assert _rel(m["tr_Sigma"], torch.trace(Sigma).item()) < TOL
    assert _rel(m["fro_Sigma_sq"], (Sigma * Sigma).sum().item()) < TOL
    assert _rel(m["tr_Omega"], torch.trace(Omega).item()) < TOL
    assert _rel(m["fro_Omega_sq"], (Omega * Omega).sum().item()) < TOL

    # ||G||_F^2 = tr(KM)
    assert _rel(m["g2"], (G * G).sum().item()) < TOL

    # PR(Sigma) = (tr Sigma)^2 / tr(Sigma^2)
    pr = torch.trace(Sigma).item() ** 2 / (Sigma * Sigma).sum().item()
    assert _rel(m["PR_Sigma"], pr) < TOL
    assert _rel(m["c"], 1.0 / pr) < TOL

    # rho_delta = tr(G Sigma G^T) / ||G||_F^2
    rho = torch.trace(G @ Sigma @ G.T).item() / (G * G).sum().item()
    assert _rel(m["rho_delta"], rho) < TOL

    # mu2 = tr(G Sigma^2 G^T) / ||G||_F^2
    mu2 = torch.trace(G @ Sigma @ Sigma @ G.T).item() / (G * G).sum().item()
    assert _rel(m["mu2"], mu2) < TOL

    # the full K-FAC Rayleigh quotient, with no isotropy assumption on Omega
    rayleigh = torch.trace(Omega @ G @ Sigma @ G.T).item() / (G * G).sum().item()
    assert _rel(m["rayleigh_full"], rayleigh) < TOL

    # PR(Omega)
    pr_om = torch.trace(Omega).item() ** 2 / (Omega * Omega).sum().item()
    assert _rel(m["PR_Omega"], pr_om) < TOL


def test_R_is_ratio_of_a_to_c(case):
    m = derive_metrics(compute_traces(case["K"], case["M"]), case["n"],
                       case["d_in"], case["d_out"])
    assert _rel(m["R"], m["a"] / m["c"]) < TOL


def test_R_omega_isolates_the_isotropy_error(case):
    """R_Omega must be exactly the ratio of the full Rayleigh quotient to what
    an isotropic Omega would predict."""
    m = derive_metrics(compute_traces(case["K"], case["M"]), case["n"],
                       case["d_in"], case["d_out"])
    predicted_if_isotropic = m["beta_iso"] * m["rho_delta"]
    assert _rel(m["R_Omega"], m["rayleigh_full"] / predicted_if_isotropic) < TOL


def test_isotropic_omega_gives_R_omega_one():
    """Constructed so Omega really is isotropic: R_Omega must land on 1."""
    torch.manual_seed(1)
    n, d_in, d_out = 400, 20, 16
    X = torch.randn(n, d_in, dtype=torch.float64)
    # Delta with orthonormal columns scaled equally -> Omega = s*I exactly.
    Q, _ = torch.linalg.qr(torch.randn(n, d_out, dtype=torch.float64))
    Delta = Q * math.sqrt(7.0)
    Xc = X - X.mean(0, keepdim=True)
    m = derive_metrics(compute_traces(Xc @ Xc.T, Delta @ Delta.T), n, d_in, d_out)

    assert _rel(m["R_Omega"], 1.0) < 1e-8
    assert _rel(m["PR_Omega"], d_out) < 1e-8


def test_R_omega_sym_matches_dense_computation(case):
    """R_Omega_sym = [tr(Omega G Sigma G^T)/tr(G Sigma G^T)] / [||Omega||_F^2/tr Omega]
    -- the same Rayleigh-quotient-over-own-mean construction as R, applied to
    Omega instead of Sigma. Unlike R_Omega (normalised against Omega's flat
    mean beta_iso while R is normalised against Sigma's eigenvalue-weighted
    mean rho_iso), this is magnitude-comparable to R by construction."""
    Sigma, Omega, G = case["Sigma"], case["Omega"], case["G"]
    m = derive_metrics(compute_traces(case["K"], case["M"]), case["n"],
                       case["d_in"], case["d_out"])

    tr_OGSG = torch.trace(Omega @ G @ Sigma @ G.T).item()
    tr_GSG = torch.trace(G @ Sigma @ G.T).item()
    fro_Omega_sq = (Omega * Omega).sum().item()
    tr_Omega = torch.trace(Omega).item()
    dense = (tr_OGSG / tr_GSG) / (fro_Omega_sq / tr_Omega)

    assert _rel(m["R_Omega_sym"], dense) < TOL


def test_isotropic_omega_gives_R_omega_sym_one():
    """Same isotropic construction as R_Omega's own test: R_Omega_sym must
    also land on exactly 1, since an isotropic Omega has no anisotropy for
    either normalisation convention to disagree about."""
    torch.manual_seed(1)
    n, d_in, d_out = 400, 20, 16
    X = torch.randn(n, d_in, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(n, d_out, dtype=torch.float64))
    Delta = Q * math.sqrt(7.0)
    Xc = X - X.mean(0, keepdim=True)
    m = derive_metrics(compute_traces(Xc @ Xc.T, Delta @ Delta.T), n, d_in, d_out)

    assert _rel(m["R_Omega_sym"], 1.0) < 1e-8


def test_R_omega_sym_is_scale_invariant(case):
    """Like R, R_Omega_sym must be invariant to a global rescaling of X or
    Delta -- it is a shape statistic, not a magnitude one."""
    K, M = case["K"], case["M"]
    base = derive_metrics(compute_traces(K, M), case["n"], case["d_in"], case["d_out"])
    scaled = derive_metrics(compute_traces(K * 13.0, M * 0.07), case["n"],
                            case["d_in"], case["d_out"])
    assert _rel(scaled["R_Omega_sym"], base["R_Omega_sym"]) < 1e-8


def test_gram_centering_flag(case):
    K_centered = gram(case["X"], center=True)
    K_raw = gram(case["X"], center=False)
    assert torch.allclose(K_centered, case["Xc"] @ case["Xc"].T)
    assert torch.allclose(K_raw, case["X"] @ case["X"].T)
    assert not torch.allclose(K_centered, K_raw)


def test_shuffle_null_preserves_marginals_and_kills_alignment(case):
    """The null must leave tr K and the gradient-magnitude marginal alone."""
    K, M = case["K"], case["M"]
    g = torch.Generator().manual_seed(7)
    tr = compute_traces(K, M)
    null = shuffle_null(K, M, g)

    # trM is invariant under a symmetric permutation, so the error-magnitude
    # distribution is untouched; only the pairing changed.
    p = torch.randperm(case["n"], generator=torch.Generator().manual_seed(7))
    Ms = M[p][:, p]
    assert _rel(torch.trace(Ms).item(), torch.trace(M).item()) < TOL
    assert set(null) == {"a_shuffled", "g2_shuffled", "rho_delta_shuffled"}
    assert math.isfinite(null["a_shuffled"])
    assert null["g2_shuffled"] != tr["trKM"]


def test_alignment_null_detects_planted_alignment():
    """With Delta built to align with the top input direction, `a` must exceed
    the shuffled null; with independent Delta it must not."""
    torch.manual_seed(3)
    n, d = 300, 24
    U = torch.randn(n, d, dtype=torch.float64)
    # Strongly anisotropic inputs.
    scales = torch.logspace(0, -2, d, dtype=torch.float64)
    X = U * scales
    Xc = X - X.mean(0, keepdim=True)
    K = Xc @ Xc.T
    g = torch.Generator().manual_seed(11)

    aligned = (Xc[:, :1] * 3.0).repeat(1, 8)          # Delta follows top input dir
    independent = torch.randn(n, 8, dtype=torch.float64)

    for Delta, expect_excess in ((aligned, True), (independent, False)):
        M = Delta @ Delta.T
        a = derive_metrics(compute_traces(K, M), n, d, 8)["a"]
        nulls = [shuffle_null(K, M, g)["a_shuffled"] for _ in range(20)]
        excess = a / (sum(nulls) / len(nulls))
        if expect_excess:
            assert excess > 1.5, f"planted alignment not detected (excess={excess:.3f})"
        else:
            assert 0.5 < excess < 2.0, f"spurious alignment (excess={excess:.3f})"


class TestFiniteNBias:
    """The finite-N floor is the reason PR must never be read at face value."""

    def test_mp_null_matches_empirical_isotropic_pr(self):
        torch.manual_seed(5)
        for n, d in ((1024, 1536), (1024, 8192), (2048, 2048)):
            X = torch.randn(n, d, dtype=torch.float64)
            eye = torch.eye(n, dtype=torch.float64)
            m = derive_metrics(compute_traces(gram(X, center=True), eye), n, d, 1)
            # Within a few percent: this is a finite-size expectation, not an
            # identity.
            assert _rel(m["PR_Sigma"], mp_null_pr(n, d)) < 0.05

    def test_pr_is_capped_near_N(self):
        """An isotropic D=8192 covariance sampled at N=1024 reports ~910, not
        8192. Any claim about dimension made at that N is measuring N."""
        assert mp_null_pr(1024, 8192) < 1000
        assert mp_null_pr(1024, 2048) < 700

    def test_dimension_contrast_collapses_at_small_N(self):
        """SmolLM2's true 4x down_proj/o_proj input-dimension contrast measures
        as ~1.3x at N=1024. This is the flaw that n_tokens_sweep exists for."""
        true_ratio = 8192 / 2048
        measured = mp_null_pr(1024, 8192) / mp_null_pr(1024, 2048)
        assert true_ratio == pytest.approx(4.0)
        assert measured < 1.5

        # and it recovers as N grows
        assert mp_null_pr(65536, 8192) / mp_null_pr(65536, 2048) > 3.0


def test_ratios_are_invariant_to_feature_scale(case):
    """c, a, R and R_Omega are defined so the (N-1) factors and any global
    rescaling of X or Delta cancel. A scale-dependent ratio would drift with
    activation norm rather than with geometry."""
    K, M = case["K"], case["M"]
    base = derive_metrics(compute_traces(K, M), case["n"], case["d_in"], case["d_out"])
    scaled = derive_metrics(compute_traces(K * 13.0, M * 0.07), case["n"],
                            case["d_in"], case["d_out"])

    for key in ("c", "PR_Sigma", "a", "R", "PR_Omega"):
        assert _rel(scaled[key], base[key]) < 1e-8, key

    # R_Omega carries one factor of Delta's scale through beta_iso = trM/D_out,
    # so it is deliberately NOT scale free; assert the known scaling instead.
    assert _rel(scaled["R_Omega"], base["R_Omega"]) < 1e-8


def test_zero_gradient_yields_nan_not_crash():
    """A module whose gradient is exactly zero must produce NaN, not a
    ZeroDivisionError that takes down the training run."""
    n = 32
    K = torch.eye(n, dtype=torch.float64)
    M = torch.zeros(n, n, dtype=torch.float64)
    m = derive_metrics(compute_traces(K, M), n, 8, 8)
    assert math.isnan(m["a"])
    assert math.isnan(m["R_Omega"])
    assert m["g2"] == 0.0


class TestTopEigenvalues:
    """Top-k spectra are a deep-probe diagnostic, but they must still be right,
    and both methods must agree on the regime real activations live in."""

    def _anisotropic(self, n, d, alpha, seed=0):
        torch.manual_seed(seed)
        X = torch.randn(n, d, dtype=torch.float64) * torch.logspace(
            0, -alpha, d, dtype=torch.float64
        )
        return X - X.mean(0, keepdim=True)

    def test_exact_matches_a_direct_eigendecomposition(self):
        Xc = self._anisotropic(120, 40, 2.0)
        ref = torch.linalg.eigvalsh(Xc @ Xc.T).flip(0)[:16]
        got = top_eigenvalues(Xc, 16, method="exact")
        assert torch.allclose(got.double(), ref, rtol=1e-6)

    def test_exact_uses_the_smaller_gram_form(self):
        """Non-zero spectra of F F^T and F^T F coincide, so the D x D form must
        give the same answer as the N x N one when D < N."""
        Xc = self._anisotropic(200, 30, 1.5)
        ref = torch.linalg.eigvalsh(Xc @ Xc.T).flip(0)[:10]
        assert torch.allclose(top_eigenvalues(Xc, 10, method="exact").double(), ref, rtol=1e-6)

    def test_randomized_tracks_exact_on_an_anisotropic_spectrum(self):
        """The regime the default is tuned for."""
        Xc = self._anisotropic(512, 512, 2.0)
        ref = top_eigenvalues(Xc, 64, method="exact")
        got = top_eigenvalues(Xc, 64, method="randomized", oversample=192, niter=8)
        assert ((got - ref).abs() / ref).max() < 1e-3

    def test_randomized_is_weaker_on_a_flat_spectrum(self):
        """Documented limitation, asserted so it cannot silently regress into
        being treated as exact."""
        Xc = self._anisotropic(512, 512, 0.0)
        ref = top_eigenvalues(Xc, 64, method="exact")
        got = top_eigenvalues(Xc, 64, method="randomized", oversample=96, niter=4)
        assert ((got - ref).abs() / ref).max() > 1e-3

    def test_eigenvalues_are_descending_and_non_negative(self):
        Xc = self._anisotropic(200, 64, 1.0)
        for method in ("exact", "randomized"):
            eig = top_eigenvalues(Xc, 32, method=method)
            assert (eig[:-1] >= eig[1:] - 1e-8).all(), method
            assert (eig >= -1e-8).all(), method

    def test_eigenvalue_sum_is_bounded_by_the_trace(self):
        """Sum of the top k eigenvalues cannot exceed tr K."""
        Xc = self._anisotropic(200, 64, 1.5)
        trK = (Xc * Xc).sum().item()
        for method in ("exact", "randomized"):
            assert top_eigenvalues(Xc, 32, method=method).sum().item() <= trK * (1 + 1e-6)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="exact|randomized"):
            top_eigenvalues(torch.randn(20, 10), 4, method="lanczos")
