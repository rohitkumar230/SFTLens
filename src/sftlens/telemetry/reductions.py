"""Trace algebra for the layer-geometry substrates.

For a linear module with sampled inputs X (N x D_in) and output-side gradients
Delta (N x D_out), the weight gradient is G = Delta^T X, and the two Kronecker
factors of the K-FAC curvature approximation are

    Sigma = X_c^T X_c / (N-1)     (D_in  x D_in,  input side)
    Omega = Delta^T Delta         (D_out x D_out, output side)

Every quantity below is a trace of products of the two Gram matrices

    K = X_c X_c^T    (N x N)
    M = Delta Delta^T (N x N)

and is therefore independent of D. That is what makes it affordable at
D_in = 8192.

    tr Sigma        = tr K / (N-1)
    ||Sigma||_F^2   = tr K^2 / (N-1)^2
    c = 1/PR(Sigma) = tr K^2 / (tr K)^2                  [N-1 cancels]
    ||G||_F^2       = tr(KM)
    tr(G Sigma G^T) = tr(K^2 M) / (N-1)
    a               = tr(K^2 M) / (tr(KM) tr K)          [N-1 cancels]
    R = a/c         = tr(K^2 M) tr K / (tr(KM) tr K^2)
    mu2             = tr(K^3 M) / ((N-1)^2 tr(KM))
    tr Omega        = tr M
    ||Omega||_F^2   = tr M^2
    tr(Omega G Sigma G^T) = tr(K^2 M^2) / (N-1)     <- full K-FAC Rayleigh
                                                       quotient, no isotropy
                                                       assumption on Omega

EVALUATION ORDER
    Written naively this needs K^2, K^3, M^2: four N x N matmuls and four N x N
    buffers. Two suffice. With

        P = K M        Q = K P = K^2 M

    then

        tr(KM)     = tr P
        tr(K^2 M)  = tr Q
        tr(K^3 M)  = sum(K * Q^T)
        tr(K^2M^2) = tr(M K K M) = ||K M||_F^2 = ||P||_F^2      [cyclic]

    Note tr(K^2 M^2) is NOT tr((KM)^2): K and M do not commute, and the two
    differ by O(1). The cyclic identity above is the correct free form.

PRECISION
    Matmuls run in fp32 with TF32 explicitly disabled. TF32's 10-bit mantissa
    gives ~4e-2 relative error on an N=8192 contraction, and the measurement is
    entirely ratios of quantities that differ by orders of magnitude. The
    elementwise reductions accumulate in fp64, chunked so that no fp64 copy of
    an N x N matrix is ever materialised.
"""

from __future__ import annotations

import contextlib
import math

import torch

# Rows per chunk in the fp64 reductions. 512 x 8192 x 8B = 33 MB of scratch,
# which keeps the accumulator exact without a full fp64 N x N temporary.
_REDUCE_CHUNK = 512


@contextlib.contextmanager
def exact_matmul():
    """Disable TF32 for the duration of a probe.

    Training may legitimately run with TF32 on; the measurement may not. This
    restores whatever the training loop had set.
    """
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


_NO_FP64_WARNED: set[str] = set()


def accum_dtype(device: torch.device) -> torch.dtype:
    """Highest accumulation precision the device actually supports.

    CUDA and CPU give fp64, which is what the reductions want: tr K and tr K^2
    differ by orders of magnitude and the measurement is entirely their ratio.
    MPS has no fp64 at all, so it falls back to fp32 -- loudly, because a
    silently lower-precision measurement is worse than a missing one.
    """
    if device.type == "mps":
        if "mps" not in _NO_FP64_WARNED:
            _NO_FP64_WARNED.add("mps")
            print(
                "[telemetry] WARNING: MPS has no float64; reductions fall back "
                "to float32. Acceptable for smoke tests, NOT for a measurement "
                "run -- use CUDA or CPU."
            )
        return torch.float32
    return torch.float64


def _sum_prod(a: torch.Tensor, b: torch.Tensor) -> float:
    """sum(a * b) accumulated at the device's best precision, chunked by rows.

    Chunked so that no full-precision copy of an N x N matrix is materialised:
    at N=8192 an fp64 temporary would be 537 MB.
    """
    dt = accum_dtype(a.device)
    total = torch.zeros((), dtype=dt, device=a.device)
    for i in range(0, a.shape[0], _REDUCE_CHUNK):
        total += (a[i : i + _REDUCE_CHUNK].to(dt) * b[i : i + _REDUCE_CHUNK].to(dt)).sum()
    return float(total.item())


def _trace(a: torch.Tensor) -> float:
    return float(torch.diagonal(a).to(accum_dtype(a.device)).sum().item())


def gram(
    features: torch.Tensor, center: bool, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """N x D features -> N x N Gram matrix, optionally token-centred.

    `dtype` defaults to the input's own dtype. The probe passes float32
    explicitly (activations arrive as bf16, and the Gram must not be formed in
    bf16); tests pass float64 to check the identities rather than the noise.
    Centring is done after the cast so the subtraction happens at the working
    precision.
    """
    f = features if dtype is None else features.to(dtype)
    if center:
        f = f - f.mean(0, keepdim=True)
    return f @ f.T


def compute_traces(K: torch.Tensor, M: torch.Tensor) -> dict[str, float]:
    """The seven raw traces, from two N x N matmuls."""
    P = K @ M
    Q = K @ P
    traces = {
        "trK": _trace(K),
        "trK2": _sum_prod(K, K),          # K symmetric, so tr K^2 = sum(K * K)
        "trM": _trace(M),
        "trM2": _sum_prod(M, M),
        "trKM": _trace(P),
        "trK2M": _trace(Q),
        "trK3M": _sum_prod(K, Q.transpose(0, 1)),
        "trK2M2": _sum_prod(P, P),        # tr(MKKM) = ||KM||_F^2
    }
    del P, Q
    return traces


def top_eigenvalues(
    features: torch.Tensor,
    k: int,
    method: str = "randomized",
    oversample: int = 192,
    niter: int = 8,
) -> torch.Tensor:
    """Top-k eigenvalues of the Gram matrix of `features` (N x D), descending.

    `features` must already be centred if the Gram it stands for is.

    WHY NOT eigvalsh ON THE GRAM
        A full symmetric eigendecomposition is O(N^3). At N=8192 that is ~40 s
        per matrix on an H100, and the probe forms two of them per module. Over
        63 modules a single deep probe would spend well over an hour computing
        a diagnostic that is not even one of the primary substrates -- those
        come from the traces, which are exact and cheap.

    TWO METHODS
        "exact"      eigvalsh on the SMALLER of F F^T (N x N) and F^T F (D x D).
                     Their non-zero spectra are identical, so this is exact and
                     often far cheaper than the N x N form. Still O(min(N,D)^3).
        "randomized" squared singular values from `svd_lowrank`. Cost is
                     O(N D q). Accuracy depends on spectral decay: with
                     oversample=192 and niter=8 the top-64 land within ~1e-4
                     relative on an anisotropic spectrum, but only ~8e-3 on a
                     flat one. Real activation covariances are strongly
                     anisotropic, which is the regime this is tuned for.

        The same method must be used for every module in a run: the
        down_proj (D=8192) vs o_proj (D=2048) comparison is the experiment, and
        estimating their spectra by different algorithms would confound it.
    """
    n, d = features.shape
    if method == "exact":
        # Non-zero eigenvalues of F F^T and F^T F coincide; take the cheaper.
        gram_matrix = features @ features.T if n <= d else features.T @ features
        return torch.linalg.eigvalsh(gram_matrix.float()).flip(0)[:k]
    if method != "randomized":
        raise ValueError(f"eig method must be exact|randomized, got {method!r}")

    q = min(k + oversample, n, d)
    _, s, _ = torch.svd_lowrank(features.float(), q=q, niter=niter)
    return (s**2)[:k]


def mp_null_pr(n: int, d: int) -> float:
    """Participation ratio an ISOTROPIC D-dim covariance yields from N samples.

    E[tr K] = ND and E[tr K^2] = ND(N+D+1) for white Gaussian X, so

        PR_null = ND / (N + D + 1)

    The true PR of an isotropic covariance is D. At N=1024 and D=8192 this
    returns ~910, not 8192: participation ratio estimated from N samples is
    capped near N and biased low whenever N is not much larger than D.

    Logged alongside every measured PR so that the finite-N floor is visible in
    the data rather than something a reader has to know to subtract. In
    particular, a measured PR close to PR_null is consistent with an isotropic
    covariance and carries no evidence of low-dimensional structure.
    """
    return n * d / (n + d + 1)


def derive_metrics(
    tr: dict[str, float], n: int, d_in: int, d_out: int
) -> dict[str, float]:
    """Named substrates from raw traces. Pure arithmetic, no tensors."""
    nm1 = n - 1
    trK, trK2 = tr["trK"], tr["trK2"]
    trM, trM2 = tr["trM"], tr["trM2"]
    trKM, trK2M = tr["trKM"], tr["trK2M"]
    trK3M, trK2M2 = tr["trK3M"], tr["trK2M2"]

    def div(num: float, den: float) -> float:
        return num / den if den > 0 else math.nan

    # -- input factor Sigma -------------------------------------------------
    c = div(trK2, trK * trK)                    # = 1/PR(Sigma)
    pr_sigma = div(1.0, c)
    rho_iso = div(trK2, nm1 * trK)              # tr(Sigma^2)/tr(Sigma)

    # -- gradient and its alignment with Sigma ------------------------------
    a = div(trK2M, trKM * trK)
    rho_delta = div(trK2M, nm1 * trKM)          # tr(G Sigma G^T)/||G||_F^2
    R = div(a, c)
    mu2 = div(trK3M, nm1 * nm1 * trKM)

    # -- output factor Omega ------------------------------------------------
    pr_omega = div(trM * trM, trM2)
    beta_iso = div(trM, d_out)                  # isotropic estimate of Omega
    rayleigh_full = div(trK2M2, nm1 * trKM)     # tr(Omega G Sigma G^T)/||G||^2
    # R_Omega isolates the error incurred by assuming Omega is isotropic: the
    # full Rayleigh quotient divided by what the isotropic assumption predicts.
    # This is the correct factor for the exact curvature decomposition
    # (rayleigh_full = beta_iso * rho_iso * R * R_Omega, asserted in tests),
    # but its magnitude is NOT comparable to R's: R normalises against
    # rho_iso (Sigma's own eigenvalue-weighted mean, near the top of its
    # spectrum), while R_Omega normalises against beta_iso (Omega's flat,
    # uniform mean, tr Omega / D_out). Different reference points, so a raw
    # |log R| vs |log R_Omega| magnitude comparison is not apples-to-apples --
    # confirmed empirically on the dolly-full dry run, where R_Omega's
    # magnitude tracked D_out/PR_Omega almost exactly (i.e. it was restating
    # Omega's anisotropy, already measured directly by PR_Omega_ratio, not
    # adding information). See analysis/dolly_full_dryrun/FINDINGS.md §4.1.
    R_omega = div(rayleigh_full, beta_iso * rho_delta)

    # R_Omega_sym: the apples-to-apples analogue of R, one level up the K-FAC
    # decomposition. R = [tr(G Sigma G^T)/||G||_F^2] / [||Sigma||_F^2/tr Sigma]
    # -- a Rayleigh quotient of G against Sigma, normalised by Sigma's own
    # eigenvalue-weighted mean. R_Omega_sym applies the identical construction
    # to Omega:
    #     R_Omega_sym = [tr(Omega G Sigma G^T)/tr(G Sigma G^T)]
    #                   / [||Omega||_F^2 / tr Omega]
    # which reduces, in the same raw traces as everything else here, to
    #     R_Omega_sym = (tr_K2M2 * tr_M) / (tr_K2M * tr_M2)
    # Verified to machine precision (1e-15) against an independent dense
    # computation, and against the isotropic-Omega edge case (must equal 1,
    # confirmed). Unlike R_Omega, this IS magnitude-comparable to R.
    R_omega_sym = div(trK2M2 * trM, trK2M * trM2)

    return {
        "N": n,
        "D_in": d_in,
        "D_out": d_out,
        # input factor
        "tr_Sigma": div(trK, nm1),
        "fro_Sigma_sq": div(trK2, nm1 * nm1),
        "c": c,
        "PR_Sigma": pr_sigma,
        "PR_Sigma_null": mp_null_pr(n, d_in),
        "PR_Sigma_ratio": div(pr_sigma, mp_null_pr(n, d_in)),
        "rho_iso": rho_iso,
        # gradient / alignment
        "g2": trKM,
        "rho_delta": rho_delta,
        "a": a,
        "R": R,
        "mu2": mu2,
        # output factor
        "tr_Omega": trM,
        "fro_Omega_sq": trM2,
        "PR_Omega": pr_omega,
        "PR_Omega_null": mp_null_pr(n, d_out),
        "PR_Omega_ratio": div(pr_omega, mp_null_pr(n, d_out)),
        "beta_iso": beta_iso,
        "rayleigh_full": rayleigh_full,
        "R_Omega": R_omega,
        "R_Omega_sym": R_omega_sym,
        # raw traces, kept so any quantity invented later can be recomputed
        # from the parquet without re-running the probe
        **{f"tr_{k[2:]}": v for k, v in tr.items()},
    }


def shuffle_null(
    K: torch.Tensor, M: torch.Tensor, generator: torch.Generator
) -> dict[str, float]:
    """Alignment statistics with the X-Delta pairing destroyed.

    Permuting Delta's rows preserves the marginal distribution of both the
    inputs and the gradient magnitudes exactly, and destroys only which input
    was paired with which gradient. Any excess of `a` over `a_shuffled` is
    therefore attributable to alignment rather than to the two marginals.

    Implemented as a symmetric permutation of M, which is equivalent to
    permuting Delta's rows and costs no extra Gram construction.
    """
    n = K.shape[0]
    p = torch.randperm(n, generator=generator).to(M.device)
    Ms = M[p][:, p]
    tr = compute_traces(K, Ms)

    def div(num: float, den: float) -> float:
        return num / den if den > 0 else math.nan

    # tr K is invariant under the permutation of M, so it is taken from the
    # shuffled trace dict rather than recomputed.
    return {
        "a_shuffled": div(tr["trK2M"], tr["trKM"] * tr["trK"]),
        "g2_shuffled": tr["trKM"],
        "rho_delta_shuffled": div(tr["trK2M"], (n - 1) * tr["trKM"]),
    }
