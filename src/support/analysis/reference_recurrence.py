"""B0-7: plain-PyTorch Mamba-3 reference recurrence. CANDIDATE CPU oracle.

NAIVE AND SEQUENTIAL ON PURPOSE. No chunking, no segsum: a token-by-token loop
is the ground truth that a chunked kernel is an optimization OF.

STATUS: candidate, NOT validated. Kernel equivalence is blocked until GPU tensor
parity (gpu_probe g2/g3). Nothing here may be cited as agreeing with the kernel.

DO NOT use upstream/hrsvrn_full/mamba3-minimal/mamba3.py as a reference. It
declares a STATIC `A_log` and computes A = -exp(A_log) (Mamba-2 style). The
official release has no static A and no A_log tensor in its checkpoints.

--------------------------------------------------------------------------
EVERY RULE BELOW IS TRANSCRIBED FROM PINNED SOURCE. Line references:
  M = upstream/pypi_232_post1/mamba_ssm/modules/mamba3.py
  K = upstream/tilelang/mamba3/mamba3_mimo_fwd.py
  G = upstream/triton/mamba3/angle_dt.py          (byte-identical to the
      pypi_base copy; verified with diff)
  N = upstream/hrsvrn_full/mamba-og/mamba_ssm/ops/triton/layernorm_gated.py

1. SPLIT ORDER                                                     M L177-190
   [z, x, B, C, dd_dt, dd_A, trap, angles]; B/C reshape to (r, g, n).

2. DISCRETISATION                                                  M L194-198
   Delta = softplus(dd_dt + dt_bias)
   A     = -heavy_tail_activation(dd_A), clamped <= -A_floor
   ADT   = A * Delta          (log-retention per step)

3. PHASE  (this is a full re-derivation; the previous version was wrong)  G L94-117
   increment_t = tanh(raw_angle_t) * PI * Delta_t
   phase_t     = ( inclusive_cumsum(increment)_t + carried_state ) mod 2*PI
   The cumsum is INCLUSIVE: position t includes its own increment. The modulo
   uses the floor convention (G L108: x - 2PI*floor(x/2PI)).
   Persistent `angle_dt_state` of shape (batch, nheads, num_rope_angles)
   (M L447) is what makes the phase cumulative ACROSS chunks and steps.

4. ROTARY ON BOTH SIDES -- ESTABLISHED FROM THE PREFILL PATH, NOT INFERRED
   Exact prefill ordering (all line numbers in K):
     L211-215  q loaded, bias added -> q_shared            UNROTATED
     L226      gemm(q_shared, k_shared) -> qk_dot_full     UNROTATED x UNROTATED
     L246-259  q_shared rotated IN PLACE by angles_frag
     L263      gemm(q_shared[ROTATED], states_accum)       interchunk
     L266-275  k_shared rotated IN PLACE, same angle, same sign as q
     L283-287  k scaled by trap_scale
     L290      gemm(q_shared[ROT], k_shared[ROT+trap])     intrachunk
   So PREFILL rotates BOTH sides with the SAME angle and the SAME sign
   (cos*x1 - sin*x2 / sin*x1 + cos*x2, identical form at L258-259 and L274-275).
   Because gemm uses transpose_B, the product is dot(R(t_i)q_i, R(t_j)k_j) =
   q_i^T R(t_j - t_i) k_j, i.e. relative -- but that is a CONSEQUENCE, not the
   reason we rotate both.
   The DIAGONAL is built at L226, BEFORE either rotation, so it uses the
   unrotated product. That is why this file computes the diagonal from Ch/Bh
   rather than Cq/Bq.
   DECODE agrees (M L364-366, apply_rotary_qk_inference_fwd(q=C, k=B)) but the
   file implementing it is absent from the pypi_232_post1 tree; prefill is the
   primary evidence and decode is corroboration only.
   PARTIAL: rotary_dim_divisor = int(2/rope_fraction) (M L99), so only
   Kang = d_state // divisor pairs rotate, pairing index n with n + d_state//2
   (K L246-251, L266-271). With rope_fraction 0.5 that is 32 pairs of 128, i.e.
   indices [0,32) against [64,96); [32,64) and [96,128) pass through untouched.

5. TRAPEZOID, AND THE TWO DISTINCT COEFFICIENTS                    K L182-198
   lambda_t      = sigmoid(trap_t)      (raw logit in; kernel applies sigmoid)
   gamma_t       = Delta_t * lambda_t
   shifted_t     = Delta_{t+1} * (1 - lambda_{t+1}),  0 at the final position
   trap_scale_t  = gamma_t + shifted_t
   These are NOT interchangeable:
     * STRICTLY EARLIER contributions use K scaled by trap_scale   K L283-287
     * the CURRENT-TOKEN DIAGONAL uses gamma ALONE                 K L326
   The kernel masks the diagonal out of the intrachunk product (K L296-298,
   "we do indeed want to exclude the diagonal") and re-adds it separately with
   gamma. Folding the diagonal into the earlier-state update is WRONG.

6. D FEEDTHROUGH IS RANK-EXPANDED                                  K L328-335
   D is added to the DIAGONAL term as  D[h] * PsiV, where PsiV is the
   rank-expanded value  V[r,p] = x[p] * mimo_x[h,r,p]  (K L206-207).
   So D multiplies the per-rank value, NOT the raw x, and it enters BEFORE the
   gate and BEFORE the mimo_o collapse.

7. GATE                                                            K L346-349
   o_gated = z[p] * mimo_z[h,r,p] * 0.5
   gate    = o_gated * tanh(o_gated) + o_gated
   Algebraically that is exactly silu(z * mimo_z): with v = z*mimo_z and
   u = v/2,  u*tanh(u)+u = (v/2)(1+tanh(v/2)) = (v/2)*2*sigmoid(v) = v*sigmoid(v).
   Verified numerically in test_gate_identity, not assumed.

8. COLLAPSE                                                        K L351-365
   y_r *= mimo_o[h,r,p] * gate_r, then SUM over r. Output is (T, H, P).

--------------------------------------------------------------------------
TWO DIFFERENT STATE OBJECTS. DO NOT CONFUSE THEM.

The trapezoid makes the naive accumulator FUTURE-AWARE: token i's stored key is
scaled by trap_scale_i = gamma_i + Delta_{i+1}(1 - lambda_{i+1}), which reads
token i+1. So an accumulator measured AFTER injecting token t depends on token
t+1 and is not a causal state. This file therefore exposes both, named apart:

  prefill_accum / prefill_accum_norms      FUTURE-AWARE. Value after injecting
      token t, so index t depends on token t+1. This is a prefill factorization
      artifact. PROHIBITED for state-trajectory claims (C-5) and for anything
      described as "the state at time t".

  online_state_norms                       CAUSAL. The state as READ at step t,
      i.e. after decay and before injecting token t. It contains tokens <= t-1
      whose trapezoidal coefficients are all fully resolved by token t. Index t
      depends only on tokens <= t. This is the object a decode implementation
      would hold, and the only one admissible for state-trajectory claims.

  pending_v[t], pending_k[t]               The deferred half of the two-token
      bookkeeping: token t's value and ROTATED, PRE-trap key, held back because
      trap_scale_t is not yet computable at step t. This mirrors the kernel
      caching k_state and v_state alongside ssm_state (M L238-239, L270-271).

  Contents, for tokens strictly before t:
      online_state[h, p, n] = sum_{i < t} exp( sum_{j in (i, t]} ADT_j )
                                * sum_r V_i[r, p] * Kbar_i[r, n]
  V_i is the rank-expanded value (x * mimo_x); Kbar_i is post-norm, post-bias,
  ROTATED and trap_scale-scaled. The rank axis is CONTRACTED, so the shape is
  (nheads, headdim, d_state) for both arms -- the documented MIMO property that
  state shape is unchanged from SISO.

  NEITHER object is asserted to equal the kernel's `ssm_state`. The kernel
  accumulates chunkwise with DA_CS_REV scaling (K L388-394) and its final-state
  convention is unverified here. Any comparison requires GPU tensor parity.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from mamba3_core import (  # noqa: E402
    InProjSpec,
    heavy_tail_activation,
    rms_norm,
    split_in_proj,
)

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# phase
# --------------------------------------------------------------------------


def phase_from_raw(raw_angle: torch.Tensor, delta: torch.Tensor,
                   carried: torch.Tensor | None = None):
    """Cumulative rotary phase. G L94-117.

    raw_angle: (T, K)      the raw `angles` slice of in_proj
    delta:     (T, H)      softplus(dd_dt + dt_bias)
    carried:   (H, K)      phase carried in from previous chunks, or None

    Returns (phase, carry_out):
      phase     (T, H, K) reduced mod 2*pi with the floor convention (G L108)
      carry_out (H, K)    running total to hand to the next chunk, itself
                          reduced mod 2*pi (G L114-117)

    Delta is per HEAD and the raw angle is shared across heads (M L202 expands
    it), so the increment is per (head, angle-pair).

    Splitting is exact because (a mod 2pi + b) mod 2pi == (a + b) mod 2pi, which
    is what test_split_sequence_phase_matches verifies.
    """
    d = phase_details(raw_angle, delta, carried)
    return d["wrapped"], d["carry_out"]


def phase_details(raw_angle: torch.Tensor, delta: torch.Tensor,
                  carried: torch.Tensor | None = None) -> dict:
    """Full source-derived phase construction. G L94-117.

    This is the SHARED implementation. `phase_from_raw` delegates here and
    returns its historical two-value result unchanged, so existing callers are
    untouched while new callers can obtain the increment and the unwrapped
    cumulative phase without transcribing the formula a second time.

    Returns:
      increment  (T, H, K)  tanh(raw) * pi * Delta
      unwrapped  (T, H, K)  inclusive cumsum, plus any carried phase
      wrapped    (T, H, K)  unwrapped mod 2*pi, floor convention (G L108)
      carry_out  (H, K)     running total for the next chunk, mod 2*pi (G L114-117)
    """
    inc = torch.tanh(raw_angle).unsqueeze(1) * math.pi * delta.unsqueeze(-1)
    unwrapped = torch.cumsum(inc, dim=0)               # INCLUSIVE (G L104)
    if carried is not None:
        unwrapped = unwrapped + carried.unsqueeze(0)
    wrapped = unwrapped - TWO_PI * torch.floor(unwrapped / TWO_PI)   # G L108
    carry = inc.sum(dim=0) + (carried if carried is not None else 0.0)
    carry = carry - TWO_PI * torch.floor(carry / TWO_PI)             # G L114-117
    return {"increment": inc, "unwrapped": unwrapped, "wrapped": wrapped,
            "carry_out": carry}


def apply_rope_split_half(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Partial split-half rotary used by the MIMO TileLang kernel. K L271-275.

    x:      (..., N)   last dim is d_state
    angles: (..., K)   K = N // rotary_dim_divisor, and K <= N//2

    For n in [0, K):
        out[..., n]        = cos*x[..., n] - sin*x[..., N//2 + n]
        out[..., N//2 + n] = sin*x[..., n] + cos*x[..., N//2 + n]
    Every other index passes through UNROTATED. Rotating all pairs is wrong.
    """
    n = x.shape[-1]
    half = n // 2
    k = angles.shape[-1]
    if k > half:
        raise ValueError(f"{k} angles but only {half} pairs available in d_state {n}")
    x1 = x[..., :k]
    x2 = x[..., half:half + k]
    c, s = torch.cos(angles), torch.sin(angles)
    out = x.clone()
    out[..., :k] = c * x1 - s * x2
    out[..., half:half + k] = s * x1 + c * x2
    return out


def apply_rope_adjacent_pairs(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Partial adjacent-pair rotary used by the SISO Triton kernel.

    The SISO kernel reshapes the final dimension as ``(..., N // 2, 2)`` and
    rotates pair ``(2*k, 2*k+1)`` for each available angle. This is not the
    MIMO split-half layout. Same-token dot products cannot reveal the mismatch
    because either simultaneous orthogonal rotation preserves them; cross-token
    reads can and do diverge.
    """
    n = x.shape[-1]
    if n % 2:
        raise ValueError(f"adjacent-pair RoPE requires even d_state, got {n}")
    k = angles.shape[-1]
    if k > n // 2:
        raise ValueError(f"{k} angles but only {n // 2} adjacent pairs available")
    pairs = x.reshape(*x.shape[:-1], n // 2, 2)
    c, s = torch.cos(angles), torch.sin(angles)
    out = pairs.clone()
    x0, x1 = pairs[..., :k, 0], pairs[..., :k, 1]
    out[..., :k, 0] = c * x0 - s * x1
    out[..., :k, 1] = s * x0 + c * x1
    return out.reshape_as(x)


def official_gate(v: torch.Tensor) -> torch.Tensor:
    """K L346-349 verbatim: o=v*0.5; o*tanh(o)+o. Equals silu(v)."""
    o = v * 0.5
    return o * torch.tanh(o) + o


# --------------------------------------------------------------------------
# the recurrence
# --------------------------------------------------------------------------


def reference_block_forward(
    u: torch.Tensor,
    params: dict,
    spec: InProjSpec,
    is_mimo: bool,
    A_floor: float = 1e-4,
    carried_phase: torch.Tensor | None = None,
    validate: bool = True,
    observer=None,
) -> dict:
    """One Mamba-3 mixer block, sequential over tokens, single sequence.

    u: (T, d_model). Returns the block output and every intermediate Stage B
    measures, so oracle and capture agree on definitions by construction.

    There are no behaviour switches. Every rule is taken from pinned source; a
    disagreement with the kernel is a bug to fix here, not a flag to toggle.

    observer: optional STREAMING OBSERVATION callback. It has NO effect on the
    recurrence: when it is None (the default) the computation, the returned
    tensors, and the op sequence are byte-for-byte what they were before this
    parameter existed. When supplied it must be either a callable invoked as
    ``observer(t, payload)`` once per token, or an object with ``on_token(t,
    payload)`` and optionally ``on_final(payload)``. Every tensor in a payload
    is a DETACHED CLONE: the callback cannot replace or mutate recurrence
    values, and may retain payload tensors freely. Per-token payload keys:

      state_read            (H,P,N) CAUSAL state as READ at t: after decay,
                            before any injection; holds tokens <= t-1 with all
                            trapezoidal coefficients resolved. This is the only
                            per-token state object admissible for
                            state-trajectory claims.
      prefill_accum         (H,P,N) FUTURE-AWARE accumulator after injecting
                            token t; index t depends on token t+1 via
                            trap_scale. PROHIBITED for state-trajectory claims.
      gamma_write           (H,P,N) gamma_t * V_t (x) Bq_t: the same-token half
                            of token t's write. source_token=t,
                            resolution_token=t.
      deferred_write        (H,P,N) shifted_gamma_t * V_t (x) Bq_t: the
                            deferred trapezoidal half, requiring token t+1's
                            Delta/lambda. source_token=t, resolution_token=t+1.
                            Zero at the final position by construction.
      write_source_token    int t
      gamma_write_resolution_token     int t
      deferred_write_resolution_token  int t+1 (None at the final position)
      earlier_read          (H,R,P) strictly-earlier state read contribution
      diagonal              (H,R,P) current-token diagonal (gamma alone)
      feedthrough           (H,R,P) D feedthrough (pre-gate, rank-expanded)
      pending_k             (H,R,N) rotated, PRE-trap key Bq_t (deferred half)
      pending_v             (H,R,P) rank-expanded value V_t (deferred half)
      K_trap                (H,R,N) trap_scale-scaled key actually injected
      C_query               (H,R,N) rotated read query Cq_t
      phase                 (H,K) wrapped cumulative phase at t
      alpha, gamma, lambda, delta, shifted_gamma, trap_scale   (H,) floats

    ``on_final`` (if present) receives detached clones of gate_factor,
    y_post_gate_per_rank, y_pre_out and out, which are computed vectorized
    after the token loop; per-token slices index dim 0.
    """
    T = u.shape[0]
    H, R, N = params["B_bias"].shape
    P = spec.headdim

    if not is_mimo and R != 1:
        raise ValueError(f"is_mimo=False requires rank-1 bias, got rank {R}")
    if is_mimo and R == 1:
        raise ValueError("is_mimo=True with rank-1 bias: check the checkpoint arm")

    # --- observer normalization. PURE OBSERVATION: when observer is None the
    #     entire mechanism is inert and the computation below is unchanged. ---
    if observer is None:
        _on_token, _on_final = None, None
    elif callable(getattr(observer, "on_token", None)):
        _on_token = observer.on_token
        _on_final = getattr(observer, "on_final", None)
        if _on_final is not None and not callable(_on_final):
            raise TypeError("observer.on_final exists but is not callable")
    elif callable(observer):
        _on_token, _on_final = observer, None
    else:
        raise TypeError(
            "observer must be None, a callable(t, payload), or an object with "
            f"on_token(t, payload); got {type(observer).__name__}")

    proj = u @ params["in_proj.weight"].T
    parts = split_in_proj(proj, spec)

    x = parts["x"].reshape(T, H, P)
    z = parts["z"].reshape(T, H, P)

    # --- discretisation (M L194-198) ---
    delta = F.softplus(parts["dd_dt"].float() + params["dt_bias"].float())
    A = (-heavy_tail_activation(parts["dd_A"].float())).clamp(max=-A_floor)
    adt = A * delta
    alpha = torch.exp(adt)                                   # (T, H)

    # --- trapezoid (K L182-198) ---
    lam = torch.sigmoid(parts["trap"].float())
    gamma = delta * lam
    shifted = torch.zeros_like(gamma)
    shifted[:-1] = delta[1:] * (1.0 - lam[1:])
    trap_scale = gamma + shifted

    # --- phase, cumulative, both sides (G L94-117, M L364-366) ---
    raw_ang = parts["angles"].reshape(T, -1)
    phase, phase_carry_out = phase_from_raw(raw_ang, delta, carried_phase)
    ph = phase.unsqueeze(2).expand(T, H, R, phase.shape[-1])

    # --- B/C: RMSNorm (M L205-206) then per-head bias inside the kernel
    #     (K L214/L220), then rotary, then trap_scale on K only (K L283-287) ---
    Bn = rms_norm(parts["B"], params["B_norm.weight"])
    Cn = rms_norm(parts["C"], params["C_norm.weight"])
    Bh = Bn.movedim(-2, -3).expand(T, H, R, N) + params["B_bias"]
    Ch = Cn.movedim(-2, -3).expand(T, H, R, N) + params["C_bias"]

    rope = apply_rope_split_half if is_mimo else apply_rope_adjacent_pairs
    Bq = rope(Bh, ph)      # rotated key; arm-specific coordinate layout
    Cq = rope(Ch, ph)      # rotated query; arm-specific coordinate layout
    Kbar = Bq * trap_scale.unsqueeze(-1).unsqueeze(-1)   # earlier-state key

    # --- rank-expanded value PsiV (K L206-207) ---
    if is_mimo:
        V = torch.einsum("thp,hrp->thrp", x, params["mimo_x"])
    else:
        V = x.unsqueeze(2)                                   # (T,H,1,P)

    state = torch.zeros(H, P, N, dtype=torch.float32)
    y = torch.zeros(T, H, R, P, dtype=torch.float32)
    # the three pre-gate pathways are recorded SEPARATELY and never folded
    # together before being stored; y (== y_pre_gate_per_rank) is their sum
    y_earlier_per_rank = torch.zeros(T, H, R, P, dtype=torch.float32)
    y_diagonal_per_rank = torch.zeros(T, H, R, P, dtype=torch.float32)
    y_feedthrough_per_rank = torch.zeros(T, H, R, P, dtype=torch.float32)
    online_state_norms = torch.zeros(T, H)      # CAUSAL: read at t, tokens < t
    prefill_accum_norms = torch.zeros(T, H)     # FUTURE-AWARE: after injecting t

    for t in range(T):
        # (a) decay, then read STRICTLY EARLIER contributions.
        state = state * alpha[t].view(H, 1, 1)
        # The state as READ at step t holds tokens <= t-1 only, and every one of
        # their trap_scale coefficients is resolved by token t. Causal.
        online_state_norms[t] = state.reshape(H, -1).norm(dim=-1)
        if _on_token is not None:
            # snapshot BEFORE injection: the only causal per-token state object
            _state_read_snapshot = state.detach().clone()
        y_earlier = torch.einsum("hrn,hpn->hrp", Cq[t], state)

        # (b) current-token DIAGONAL, gamma alone (K L326). Uses the UNROTATED
        #     product, which is identical to the rotated one at zero relative
        #     angle; the kernel exploits exactly this (its qk_dot is built
        #     before the rotary at K L226).
        qk = torch.einsum("hrn,hsn->hrs", Ch[t], Bh[t])
        y_diag = torch.einsum("hrs,hsp->hrp", qk, V[t]) * gamma[t].view(H, 1, 1)

        # (c) D feedthrough joins the DIAGONAL, rank-expanded (K L328-335).
        #     D multiplies the RANK-EXPANDED value V, not the raw x.
        y_feed = params["D"].view(H, 1, 1) * V[t]
        y_earlier_per_rank[t] = y_earlier
        y_diagonal_per_rank[t] = y_diag
        y_feedthrough_per_rank[t] = y_feed
        y[t] = y_earlier + y_diag + y_feed

        # (d) inject this token for FUTURE reads, trap_scale-weighted.
        #     trap_scale[t] reads token t+1, so the accumulator measured HERE is
        #     future-aware. It is exposed under a name that says so.
        state = state + torch.einsum("hrp,hrn->hpn", V[t], Kbar[t])
        prefill_accum_norms[t] = state.reshape(H, -1).norm(dim=-1)

        if _on_token is not None:
            # Streaming observation. Every tensor is a detached clone: the
            # callback cannot mutate recurrence values. The write is split into
            # its two trapezoidal halves with EXPLICIT resolution indexing:
            # gamma_write resolves at t; deferred_write needs token t+1's
            # Delta/lambda (shifted_gamma[t]) and resolves at t+1. Their sum is
            # exactly the injection above (Kbar = Bq * (gamma + shifted)).
            _w = torch.einsum("hrp,hrn->hpn", V[t], Bq[t]).detach()
            _last = (t == T - 1)
            _on_token(t, {
                "state_read": _state_read_snapshot,
                "prefill_accum": state.detach().clone(),
                "gamma_write": _w * gamma[t].view(H, 1, 1),
                "deferred_write": _w * shifted[t].view(H, 1, 1),
                "write_source_token": t,
                "gamma_write_resolution_token": t,
                "deferred_write_resolution_token": None if _last else t + 1,
                "earlier_read": y_earlier.detach().clone(),
                "diagonal": y_diag.detach().clone(),
                "feedthrough": y_feed.detach().clone(),
                "pending_k": Bq[t].detach().clone(),
                "pending_v": V[t].detach().clone(),
                "K_trap": Kbar[t].detach().clone(),
                "C_query": Cq[t].detach().clone(),
                "phase": phase[t].detach().clone(),
                "alpha": alpha[t].detach().clone(),
                "gamma": gamma[t].detach().clone(),
                "lambda": lam[t].detach().clone(),
                "delta": delta[t].detach().clone(),
                "shifted_gamma": shifted[t].detach().clone(),
                "trap_scale": trap_scale[t].detach().clone(),
            })

    # --- gate then collapse (K L346-365), every stage exposed ---
    if is_mimo:
        gate_pre_activation = torch.einsum("thp,hrp->thrp", z, params["mimo_z"])
        collapse_weight = params["mimo_o"]                       # (H, R, P)
    else:
        # SISO: singleton rank axis, and the collapse is the identity
        gate_pre_activation = z.unsqueeze(2)                     # (T, H, 1, P)
        collapse_weight = torch.ones(H, 1, P, dtype=y.dtype)

    gate_factor = official_gate(gate_pre_activation)             # (T, H, R, P)
    y_post_gate_per_rank = y * gate_factor
    y_collapse_contribution_per_rank = y_post_gate_per_rank * collapse_weight
    y_out = y_collapse_contribution_per_rank.sum(dim=2)          # (T, H, P)

    out = y_out.reshape(T, H * P) @ params["out_proj.weight"].T

    if _on_final is not None:
        # gate/collapse/out are computed vectorized after the loop; per-token
        # slices index dim 0. Detached clones: observation only.
        _on_final({
            "gate_factor": gate_factor.detach().clone(),
            "y_post_gate_per_rank": y_post_gate_per_rank.detach().clone(),
            "y_pre_out": y_out.detach().clone(),
            "out": out.detach().clone(),
        })

    if validate:
        # exact algebraic identities of the decomposition (FP32, tight)
        assert torch.allclose(
            y, y_earlier_per_rank + y_diagonal_per_rank + y_feedthrough_per_rank,
            atol=1e-6), "pre_gate != earlier + diagonal + feedthrough"
        assert torch.allclose(y_post_gate_per_rank, y * gate_factor, atol=1e-6)
        assert torch.allclose(y_collapse_contribution_per_rank,
                              y_post_gate_per_rank * collapse_weight, atol=1e-6)
        assert torch.allclose(
            y_out, y_collapse_contribution_per_rank.sum(dim=2), atol=1e-6)

    return {
        "out": out,
        "y_pre_out": y_out,
        # DEPRECATED NAME, kept for compatibility. "y_per_rank" is ambiguous:
        # it is the COMBINED PRE-GATE per-rank tensor, not a post-gate or
        # per-pathway quantity. Prefer y_pre_gate_per_rank.
        "y_per_rank": y,
        # --- pre-gate pathways, recorded separately (T, H, R, P) ---
        "y_pre_gate_per_rank": y,
        "y_earlier_per_rank": y_earlier_per_rank,
        "y_diagonal_per_rank": y_diagonal_per_rank,
        "y_feedthrough_per_rank": y_feedthrough_per_rank,
        # --- gate and collapse ---
        "gate_pre_activation": gate_pre_activation,
        "gate_factor": gate_factor,
        "y_post_gate_per_rank": y_post_gate_per_rank,
        "collapse_weight": collapse_weight,
        "y_collapse_contribution_per_rank": y_collapse_contribution_per_rank,
        # --- state objects, deliberately named apart (see module docstring) ---
        "prefill_accum": state,              # FUTURE-AWARE final accumulator
        "prefill_accum_norms": prefill_accum_norms,   # index t reads token t+1
        "online_state_norms": online_state_norms,     # CAUSAL, tokens <= t-1
        "pending_v": V,                      # deferred half of the two-token
        "pending_k": Bq,                     # bookkeeping: rotated, PRE-trap
        "phase": phase,
        "phase_carry_out": phase_carry_out,
        "Delta": delta, "A": A, "ADT": adt, "alpha": alpha,
        "lambda": lam, "gamma": gamma, "shifted_gamma": shifted,
        "trap_scale": trap_scale,
        "B_eff": Bq, "C_eff": Cq, "K_trap": Kbar, "V_rank": V,
        "feedthrough": params["D"].view(H, 1, 1) * V,
        "local_halflife": math.log(2.0) / (-adt).clamp_min(1e-12),
    }


# --------------------------------------------------------------------------
# deterministic self-tests
# --------------------------------------------------------------------------


def _toy(H=3, R=4, N=8, P=4, d_model=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    spec = InProjSpec(
        d_inner=H * P, d_state=N, ngroups=1, mimo_rank=R,
        nheads=H, headdim=P, n_rope_angles=N // 4,
    )
    params = {
        "in_proj.weight": torch.randn(spec.total, d_model, generator=g) * 0.1,
        "out_proj.weight": torch.randn(d_model, H * P, generator=g) * 0.1,
        "dt_bias": torch.randn(H, generator=g) * 0.1,
        "D": torch.randn(H, generator=g),
        "B_bias": 1 + torch.zeros(H, R, N),
        "C_bias": 1 + torch.zeros(H, R, N),
        "B_norm.weight": torch.ones(N),
        "C_norm.weight": torch.ones(N),
        "mimo_x": torch.full((H, R, P), 1.0 / R),
        "mimo_z": torch.ones(H, R, P),
        "mimo_o": torch.full((H, R, P), 1.0 / R),
    }
    return spec, params, d_model


def test_gate_identity():
    """The kernel's o*tanh(o)+o with o=v/2 must equal silu(v) exactly."""
    v = torch.linspace(-8, 8, 2001)
    err = (official_gate(v) - F.silu(v)).abs().max().item()
    assert err < 1e-5, err
    print(f"  gate identity o*tanh(o)+o == silu   max err {err:.2e}     OK")


def test_future_never_affects_past():
    """Token t must not change ANY output before t. Replaces the stale test
    that permitted token t to alter output t-1."""
    spec, params, d_model = _toy(seed=1)
    u = torch.randn(14, d_model)
    base = reference_block_forward(u, params, spec, True)["out"]
    worst = 0.0
    for t in (5, 9, 13):
        u2 = u.clone()
        u2[t] += 7.0
        got = reference_block_forward(u2, params, spec, True)["out"]
        worst = max(worst, (base[:t] - got[:t]).abs().max().item())
        assert not torch.allclose(base[t], got[t]), f"token {t} changed nothing at t"
    assert worst < 1e-6, f"future leaked into the past by {worst}"
    print(f"  future never affects past           max leak {worst:.2e}   OK")


def test_current_token_diagonal_uses_gamma():
    """Zeroing gamma at one position must change that position's output, and
    the change must scale linearly with gamma when the state is empty."""
    spec, params, d_model = _toy(seed=2)
    u = torch.randn(6, d_model)
    r = reference_block_forward(u, params, spec, True)

    # at t=0 the state is empty, so y[0] = gamma_0 * qk V + D V exactly
    g0 = r["gamma"][0].view(-1, 1, 1)
    qk = torch.einsum("hrn,hsn->hrs",
                      _unrot(params, spec, u, 0, "C"),
                      _unrot(params, spec, u, 0, "B"))
    expect = torch.einsum("hrs,hsp->hrp", qk, r["V_rank"][0]) * g0 \
        + params["D"].view(-1, 1, 1) * r["V_rank"][0]
    err = (r["y_per_rank"][0] - expect).abs().max().item()
    assert err < 1e-4, err
    print(f"  diagonal = gamma*qk*V + D*V at t=0  max err {err:.2e}     OK")


def _unrot(params, spec, u, t, which):
    """Recompute the UNROTATED post-bias B or C at position t, for test use."""
    parts = split_in_proj(u @ params["in_proj.weight"].T, spec)
    n = rms_norm(parts[which], params[f"{which}_norm.weight"])
    H, R, N = params["B_bias"].shape
    return (n.movedim(-2, -3).expand(u.shape[0], H, R, N) + params[f"{which}_bias"])[t]


def test_earlier_tokens_use_trap_scale():
    """A change in Delta_{t+1}/lambda_{t+1} must alter token t's stored key.

    trap_scale_t = gamma_t + Delta_{t+1}(1-lambda_{t+1}), so the stored key
    depends on the NEXT token. That is legal -- it only affects outputs at
    times > t -- and is the defining property of the trapezoid.
    """
    spec, params, d_model = _toy(seed=3)
    u = torch.randn(8, d_model)
    r = reference_block_forward(u, params, spec, True)

    manual = r["gamma"][:-1] + r["Delta"][1:] * (1.0 - r["lambda"][1:])
    err = (r["trap_scale"][:-1] - manual).abs().max().item()
    assert err < 1e-6, err
    assert torch.allclose(r["trap_scale"][-1], r["gamma"][-1]), "last must be gamma only"
    print(f"  trap_scale = gamma + shifted        max err {err:.2e}     OK")


def test_phase_is_cumulative_and_wrapped():
    """Phase must be an inclusive cumsum of tanh(raw)*pi*Delta, mod 2pi."""
    T, H, Kang = 9, 3, 2
    raw = torch.randn(T, Kang)
    delta = torch.rand(T, H) + 0.5
    ph, carry = phase_from_raw(raw, delta)

    inc = torch.tanh(raw).unsqueeze(1) * math.pi * delta.unsqueeze(-1)
    manual = torch.cumsum(inc, dim=0)
    manual = manual - TWO_PI * torch.floor(manual / TWO_PI)
    err = (ph - manual).abs().max().item()
    assert err < 1e-6, err
    assert (ph >= 0).all() and (ph < TWO_PI + 1e-6).all(), "not wrapped into [0,2pi)"
    assert (carry >= 0).all() and (carry < TWO_PI + 1e-6).all()

    # inclusive: position 0 already carries its own increment
    assert not torch.allclose(ph[0], torch.zeros_like(ph[0]))
    carried = torch.full((H, Kang), 1.0)
    ph2, _ = phase_from_raw(raw, delta, carried)
    assert not torch.allclose(ph, ph2)
    print(f"  phase cumulative + wrapped mod 2pi  max err {err:.2e}     OK")


def test_split_sequence_phase_matches():
    """Blocker 5: two chunks with carried phase == one pass. G L104-117.

    Exactness relies on (a mod 2pi + b) mod 2pi == (a + b) mod 2pi.
    """
    T, H, Kang, s = 11, 3, 2, 4
    raw = torch.randn(T, Kang)
    delta = torch.rand(T, H) + 0.5

    whole, whole_carry = phase_from_raw(raw, delta)
    first, carry = phase_from_raw(raw[:s], delta[:s])
    second, carry2 = phase_from_raw(raw[s:], delta[s:], carry)

    joined = torch.cat([first, second], dim=0)
    # compare on the circle: a wrap boundary must not count as a mismatch
    d = torch.remainder(joined - whole + math.pi, TWO_PI) - math.pi
    err = d.abs().max().item()
    assert err < 1e-5, err
    dc = torch.remainder(carry2 - whole_carry + math.pi, TWO_PI) - math.pi
    assert dc.abs().max().item() < 1e-5
    print(f"  split-sequence phase == single pass max err {err:.2e}     OK")


def test_online_state_is_causal():
    """Blocker 3: token t+1 must not change ANYTHING labelled online state at t.

    The naive accumulator fails this by construction, because trap_scale[t]
    reads token t+1. That object is exposed as prefill_accum_* and is expected
    to differ; this test pins the distinction rather than hiding it.
    """
    spec, params, d_model = _toy(seed=11)
    u = torch.randn(12, d_model)
    base = reference_block_forward(u, params, spec, True)

    worst_online, saw_prefill_change = 0.0, False
    for t in (3, 6, 9):
        u2 = u.clone()
        u2[t + 1] += 6.0
        got = reference_block_forward(u2, params, spec, True)
        # online state at indices <= t must be untouched
        worst_online = max(
            worst_online,
            (base["online_state_norms"][: t + 1] - got["online_state_norms"][: t + 1])
            .abs().max().item(),
        )
        if not torch.allclose(base["prefill_accum_norms"][t],
                              got["prefill_accum_norms"][t]):
            saw_prefill_change = True

    assert worst_online < 1e-6, f"online state leaked the future by {worst_online}"
    assert saw_prefill_change, (
        "prefill_accum_norms did not react to token t+1; the two objects would "
        "then be identical and the distinction meaningless"
    )
    print(f"  online state causal (prefill accum is not) leak {worst_online:.2e} OK")


def test_manual_three_token_fixture():
    """Blocker 4: hand-assembled 3-token recurrence, validating INDEXING.

    Uses the intermediates the function returns (alpha, gamma, K_trap, V_rank,
    C_eff, unrotated C/B, D) as GIVENS and re-assembles y with explicit indices.
    It deliberately does NOT recompute alpha/gamma/trap_scale from raw inputs --
    that would re-test the helper formulas instead of the loop's bookkeeping.

    Checks, per token:
      t=0  empty state  -> y = gamma_0 * (C_0.B_0) V_0 + D V_0
      t=1  one earlier  -> + C_1 . (alpha_1 * V_0 (x) Kbar_0)
      t=2  two earlier  -> + C_2 . (alpha_2*(alpha_1*V_0(x)Kbar_0 + V_1(x)Kbar_1))
    A misplaced alpha index, or using Kbar_t instead of Kbar_{t-1}, fails here.
    """
    # SISO, scalar-sized on purpose. N=8 (not 2) only because _toy derives
    # n_rope_angles = N//4 and the Stage 2A shape contract requires it to be
    # >= 1; the indexing assertions below are unaffected by d_state.
    H, R, N, P, T = 1, 1, 8, 1, 3
    spec, params, d_model = _toy(H=H, R=R, N=N, P=P, d_model=4, seed=21)
    u = torch.randn(T, d_model)
    r = reference_block_forward(u, params, spec, is_mimo=False)

    alpha, gamma = r["alpha"], r["gamma"]
    V, Kbar, Cq = r["V_rank"], r["K_trap"], r["C_eff"]
    D = params["D"].view(H, 1, 1)

    Cu = torch.stack([_unrot(params, spec, u, t, "C") for t in range(T)])
    Bu = torch.stack([_unrot(params, spec, u, t, "B") for t in range(T)])

    def diag(t):
        qk = torch.einsum("hrn,hsn->hrs", Cu[t], Bu[t])
        return torch.einsum("hrs,hsp->hrp", qk, V[t]) * gamma[t].view(H, 1, 1)

    outer = [torch.einsum("hrp,hrn->hpn", V[i], Kbar[i]) for i in range(T)]

    expect = [
        diag(0) + D * V[0],
        torch.einsum("hrn,hpn->hrp", Cq[1], alpha[1].view(H, 1, 1) * outer[0])
        + diag(1) + D * V[1],
        torch.einsum(
            "hrn,hpn->hrp", Cq[2],
            alpha[2].view(H, 1, 1) * (alpha[1].view(H, 1, 1) * outer[0] + outer[1]),
        ) + diag(2) + D * V[2],
    ]
    worst = max((r["y_per_rank"][t] - expect[t]).abs().max().item() for t in range(T))
    assert worst < 1e-5, worst

    # negative control: shifting the decay index by one must break it
    wrong = torch.einsum("hrn,hpn->hrp", Cq[2],
                         alpha[1].view(H, 1, 1) * (alpha[2].view(H, 1, 1) * outer[0]
                                                   + outer[1])) + diag(2) + D * V[2]
    assert not torch.allclose(r["y_per_rank"][2], wrong, atol=1e-5), (
        "swapping alpha_1 and alpha_2 produced the same answer; the fixture "
        "is not sensitive to decay indexing"
    )
    print(f"  manual 3-token fixture (indexing)   max err {worst:.2e}     OK")


def test_partial_rope_indexing():
    """Only the first K pairs rotate; the rest must be untouched, and the
    rotation must preserve the norm of each rotated pair."""
    N, K = 16, 4
    x = torch.randn(5, N)
    ang = torch.randn(5, K)
    y = apply_rope_split_half(x, ang)

    assert torch.allclose(y[:, K:N // 2], x[:, K:N // 2]), "low unrotated block moved"
    assert torch.allclose(y[:, N // 2 + K:], x[:, N // 2 + K:]), "high unrotated block moved"
    pair_before = x[:, :K] ** 2 + x[:, N // 2:N // 2 + K] ** 2
    pair_after = y[:, :K] ** 2 + y[:, N // 2:N // 2 + K] ** 2
    assert torch.allclose(pair_before, pair_after, atol=1e-5)
    assert torch.allclose(apply_rope_split_half(x, torch.zeros(5, K)), x, atol=1e-6)
    print("  partial rope: indexing + norm-preserving             OK")


def test_siso_adjacent_pair_rope_indexing():
    """SISO rotates adjacent pairs and must differ from MIMO split-half."""
    N, K = 16, 4
    x = torch.arange(2 * N, dtype=torch.float32).reshape(2, N) / 7
    ang = torch.full((2, K), 0.37)
    y = apply_rope_adjacent_pairs(x, ang)

    # Only coordinates belonging to the first K adjacent pairs may move.
    assert torch.allclose(y[:, 2 * K:], x[:, 2 * K:])
    pairs_before = x[:, :2 * K].reshape(2, K, 2).square().sum(-1)
    pairs_after = y[:, :2 * K].reshape(2, K, 2).square().sum(-1)
    assert torch.allclose(pairs_before, pairs_after, atol=1e-5)
    assert not torch.allclose(y, apply_rope_split_half(x, ang))
    print("  SISO adjacent-pair rope: indexing + arm distinction OK")


def test_shapes_both_arms():
    for is_mimo, rank in ((True, 4), (False, 1)):
        spec, params, d_model = _toy(R=rank)
        u = torch.randn(11, d_model)
        r = reference_block_forward(u, params, spec, is_mimo=is_mimo)
        assert r["out"].shape == (11, d_model)
        assert r["y_per_rank"].shape == (11, 3, rank, 4)
        assert r["prefill_accum"].shape == (3, 4, 8), r["prefill_accum"].shape
        assert r["online_state_norms"].shape == (11, 3)
        assert r["pending_v"].shape == (11, 3, rank, 4)
        assert r["pending_k"].shape == (11, 3, rank, 8)
        assert torch.isfinite(r["out"]).all()
    print("  shapes: MIMO and SISO, state rank-contracted         OK")


def test_arm_mismatch_rejected():
    spec, params, d_model = _toy(R=4)
    for kwargs, msg in (({"is_mimo": False}, "rank-4 as SISO"),):
        try:
            reference_block_forward(torch.randn(6, d_model), params, spec, **kwargs)
            raise AssertionError(f"{msg} should have raised")
        except ValueError:
            pass
    spec1, params1, d1 = _toy(R=1)
    try:
        reference_block_forward(torch.randn(6, d1), params1, spec1, is_mimo=True)
        raise AssertionError("rank-1 as MIMO should have raised")
    except ValueError:
        pass
    print("  arm/rank mismatch rejected                           OK")


def test_phase_from_raw_backward_compatible():
    """phase_from_raw must return EXACTLY what it returned before delegation."""
    T, H, K = 7, 3, 2
    raw = torch.randn(T, K)
    delta = torch.rand(T, H) + 0.5
    for carried in (None, torch.rand(H, K)):
        # the pre-refactor arithmetic, written out verbatim
        inc = torch.tanh(raw).unsqueeze(1) * math.pi * delta.unsqueeze(-1)
        ph = torch.cumsum(inc, dim=0)
        if carried is not None:
            ph = ph + carried.unsqueeze(0)
        ph = ph - TWO_PI * torch.floor(ph / TWO_PI)
        carry = inc.sum(dim=0) + (carried if carried is not None else 0.0)
        carry = carry - TWO_PI * torch.floor(carry / TWO_PI)

        got_ph, got_carry = phase_from_raw(raw, delta, carried)
        assert torch.equal(got_ph, ph), "wrapped phase changed"
        assert torch.equal(got_carry, carry), "carry_out changed"

        d = phase_details(raw, delta, carried)
        assert torch.equal(d["wrapped"], ph) and torch.equal(d["carry_out"], carry)
        assert torch.equal(d["increment"], inc)
        assert torch.equal(d["wrapped"],
                           d["unwrapped"] - TWO_PI * torch.floor(d["unwrapped"] / TWO_PI))
    print("  phase_from_raw bitwise unchanged; details added   OK")


def test_pathway_decomposition_identities():
    """Exact algebra of the pre-gate/gate/collapse decomposition."""
    spec, params, d_model = _toy(seed=31)
    u = torch.randn(9, d_model)
    r = reference_block_forward(u, params, spec, is_mimo=True)

    e = {}
    e["pre_gate"] = (r["y_pre_gate_per_rank"]
                     - (r["y_earlier_per_rank"] + r["y_diagonal_per_rank"]
                        + r["y_feedthrough_per_rank"])).abs().max().item()
    e["post_gate"] = (r["y_post_gate_per_rank"]
                      - r["y_pre_gate_per_rank"] * r["gate_factor"]).abs().max().item()
    e["collapse"] = (r["y_collapse_contribution_per_rank"]
                     - r["y_post_gate_per_rank"] * r["collapse_weight"]).abs().max().item()
    e["sum_rank"] = (r["y_pre_out"]
                     - r["y_collapse_contribution_per_rank"].sum(2)).abs().max().item()
    e["out"] = (r["out"] - r["y_pre_out"].reshape(9, -1)
                @ params["out_proj.weight"].T).abs().max().item()
    for k, v in e.items():
        assert v < 1e-6, (k, v)
    assert torch.equal(r["y_per_rank"], r["y_pre_gate_per_rank"]), "alias broken"
    print(f"  decomposition identities  max err "
          f"{max(e.values()):.2e}  {({k: f'{v:.1e}' for k, v in e.items()})}   OK")


def test_gate_separability():
    """The gate is multiplicative WITHIN each rank, not a nonlinearity over the
    rank sum. Changing mimo_z for rank r moves only rank r's gate_factor."""
    spec, params, d_model = _toy(seed=32)
    u = torch.randn(6, d_model)
    base = reference_block_forward(u, params, spec, is_mimo=True)

    p2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in params.items()}
    p2["mimo_z"][:, 1, :] *= 2.5                      # perturb rank 1 only
    mod = reference_block_forward(u, p2, spec, is_mimo=True)

    g0, g1 = base["gate_factor"], mod["gate_factor"]
    moved = not torch.allclose(g0[:, :, 1], g1[:, :, 1])
    others = [rr for rr in range(g0.shape[2]) if rr != 1]
    untouched = torch.allclose(g0[:, :, others], g1[:, :, others], atol=1e-7)
    collapsed_moved = not torch.allclose(base["y_pre_out"], mod["y_pre_out"])
    assert moved and untouched and collapsed_moved
    print("  gate separability: rank-1 gate moved, other ranks' gates unchanged "
          "pre-collapse, collapsed output still moved (ranks are SUMMED)   OK")


def test_feedthrough_definition():
    """D multiplies the RANK-EXPANDED value, before gate and collapse."""
    spec, params, d_model = _toy(seed=33)
    u = torch.randn(8, d_model)
    r = reference_block_forward(u, params, spec, is_mimo=True)
    H = params["D"].shape[0]

    exact = (r["feedthrough"] - params["D"].view(H, 1, 1) * r["V_rank"]).abs().max()
    assert float(exact) == 0.0
    assert torch.equal(r["y_feedthrough_per_rank"], r["feedthrough"])

    # differs across ranks when mimo_x differs
    p2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in params.items()}
    p2["mimo_x"][:, 0, :] *= 3.0
    r2 = reference_block_forward(u, p2, spec, is_mimo=True)
    rank_dep = not torch.allclose(r2["y_feedthrough_per_rank"][:, :, 0],
                                  r2["y_feedthrough_per_rank"][:, :, 1])
    # NEGATIVE CONTROL: D * raw_x (rank-independent) must NOT reproduce it
    parts = split_in_proj(u @ params["in_proj.weight"].T, spec)
    x = parts["x"].reshape(u.shape[0], H, spec.headdim)
    wrong = (params["D"].view(H, 1, 1) * x.unsqueeze(2)).expand_as(r["feedthrough"])
    fails = not torch.allclose(wrong, r["feedthrough"], atol=1e-5)
    # and it enters BEFORE the gate
    before_gate = torch.allclose(
        r["y_pre_gate_per_rank"],
        r["y_earlier_per_rank"] + r["y_diagonal_per_rank"] + r["feedthrough"],
        atol=1e-6)
    assert rank_dep and fails and before_gate
    print("  feedthrough == D * V_rank, rank-dependent, pre-gate; D*raw_x "
          "negative control FAILS as required   OK")


def test_diagonal_versus_earlier_state():
    """Diagonal: gamma only, UNROTATED. Earlier: rotated C reading trap-weighted
    rotated B. Excluding the diagonal from the accumulation avoids double count."""
    spec, params, d_model = _toy(seed=34)
    u = torch.randn(5, d_model)
    r = reference_block_forward(u, params, spec, is_mimo=True)
    H = params["D"].shape[0]

    # t=0: state is empty, so earlier == 0 and the whole pre-gate term is
    # diagonal + feedthrough
    assert float(r["y_earlier_per_rank"][0].abs().max()) == 0.0
    Cu = _unrot(params, spec, u, 0, "C")
    Bu = _unrot(params, spec, u, 0, "B")
    qk = torch.einsum("hrn,hsn->hrs", Cu, Bu)
    expect = torch.einsum("hrs,hsp->hrp", qk, r["V_rank"][0]) \
        * r["gamma"][0].view(H, 1, 1)
    err = (r["y_diagonal_per_rank"][0] - expect).abs().max().item()
    assert err < 1e-5, err

    # the diagonal uses gamma, NOT trap_scale
    wrong = torch.einsum("hrs,hsp->hrp", qk, r["V_rank"][0]) \
        * r["trap_scale"][0].view(H, 1, 1)
    assert not torch.allclose(r["y_diagonal_per_rank"][0], wrong, atol=1e-5)

    # The diagonal may be built from the UNROTATED product because the rotation
    # is ORTHOGONAL and applied identically to both sides at the SAME position:
    #   (R C) . (R B)^T = C R^T R B^T = C B^T
    # so the same-token product is rotation-invariant. That is exactly why the
    # kernel computes qk_dot at K L226, BEFORE the rotary.
    qk_rot = torch.einsum("hrn,hsn->hrs", r["C_eff"][0], r["B_eff"][0])
    rot_invariant = torch.allclose(qk, qk_rot, atol=1e-5)
    assert rot_invariant, "same-token product should be rotation-invariant"

    # no double counting: token 0 enters the state only via K_trap, and the
    # earlier-state read at t=1 must equal C_rot . (alpha_1 * V_0 (x) Kbar_0)
    outer = torch.einsum("hrp,hrn->hpn", r["V_rank"][0], r["K_trap"][0])
    expect1 = torch.einsum("hrn,hpn->hrp", r["C_eff"][1],
                           r["alpha"][1].view(H, 1, 1) * outer)
    err1 = (r["y_earlier_per_rank"][1] - expect1).abs().max().item()
    assert err1 < 1e-5, err1
    print(f"  diagonal(gamma, unrotated) err {err:.2e}; same-token product "
          f"rotation-invariant {rot_invariant} (why K L226 precedes the rotary); "
          f"earlier(rotated, trap-weighted) err {err1:.2e}   OK")


def test_siso_control_and_equivalence():
    """SISO: singleton rank, identity collapse, and byte-identical to the old form."""
    spec, params, d_model = _toy(R=1, seed=35)
    u = torch.randn(7, d_model)
    r = reference_block_forward(u, params, spec, is_mimo=False)

    for k in ("y_pre_gate_per_rank", "y_earlier_per_rank", "y_diagonal_per_rank",
              "y_feedthrough_per_rank", "y_post_gate_per_rank", "gate_factor",
              "y_collapse_contribution_per_rank"):
        assert r[k].shape[2] == 1, (k, r[k].shape)
    assert torch.equal(r["collapse_weight"], torch.ones_like(r["collapse_weight"]))
    assert torch.allclose(r["y_pre_out"],
                          r["y_collapse_contribution_per_rank"].squeeze(2), atol=1e-7)

    # the PREVIOUS SISO formulation, written out
    parts = split_in_proj(u @ params["in_proj.weight"].T, spec)
    H, P = params["D"].shape[0], spec.headdim
    z = parts["z"].reshape(u.shape[0], H, P)
    old_y_out = r["y_pre_gate_per_rank"].squeeze(2) * official_gate(z)
    old_out = old_y_out.reshape(u.shape[0], H * P) @ params["out_proj.weight"].T
    e1 = (r["y_pre_out"] - old_y_out).abs().max().item()
    e2 = (r["out"] - old_out).abs().max().item()
    assert e1 == 0.0 and e2 == 0.0, (e1, e2)
    print(f"  SISO: singleton rank, collapse_weight==1, rank-sum identity; "
          f"reproduces old y_pre_out/out EXACTLY ({e1:.0e}/{e2:.0e})   OK")


def test_observer_noop_leaves_returns_identical():
    """Enabling a no-op observer must leave EVERY returned tensor bitwise
    identical to the observer-free run, for both arms."""
    for is_mimo, rank in ((True, 4), (False, 1)):
        spec, params, d_model = _toy(R=rank, seed=41)
        u = torch.randn(13, d_model)
        base = reference_block_forward(u, params, spec, is_mimo=is_mimo)

        class _Sink:
            def __init__(self):
                self.tokens = []
                self.final = None
            def on_token(self, t, payload):
                self.tokens.append((t, payload))
            def on_final(self, payload):
                self.final = payload

        sink = _Sink()
        obs = reference_block_forward(u, params, spec, is_mimo=is_mimo,
                                      observer=sink)
        for k in base:
            if torch.is_tensor(base[k]):
                assert torch.equal(base[k], obs[k]), f"{k} changed under observer"
            else:
                assert base[k] == obs[k], f"{k} changed under observer"
        assert len(sink.tokens) == 13 and sink.final is not None
        # plain-callable form is also accepted
        seen = []
        obs2 = reference_block_forward(u, params, spec, is_mimo=is_mimo,
                                       observer=lambda t, p: seen.append(t))
        assert torch.equal(base["out"], obs2["out"]) and seen == list(range(13))
    print("  observer: no-op callback leaves all return tensors bitwise "
          "identical; both arms, both callback forms   OK")


def test_observer_payload_semantics():
    """Payload tensors are detached clones; mutating them cannot change the
    recurrence; write halves and resolution indices are exact."""
    spec, params, d_model = _toy(R=4, seed=42)
    T = 9
    u = torch.randn(T, d_model)
    base = reference_block_forward(u, params, spec, is_mimo=True)
    H = params["D"].shape[0]

    payloads = {}

    def grab(t, p):
        payloads[t] = p
        # hostile mutation: must NOT touch the recurrence
        p["state_read"].fill_(1e9)
        p["prefill_accum"].fill_(-1e9)
        p["gamma_write"].zero_()

    got = reference_block_forward(u, params, spec, is_mimo=True, observer=grab)
    assert torch.equal(base["out"], got["out"]), "observer mutated recurrence"
    assert torch.equal(base["prefill_accum"], got["prefill_accum"])

    # re-run with a passive observer to inspect uncorrupted payloads
    payloads.clear()
    reference_block_forward(u, params, spec, is_mimo=True,
                            observer=lambda t, p: payloads.__setitem__(t, p))
    for t, p in payloads.items():
        assert not p["state_read"].requires_grad
        # gamma + deferred halves sum EXACTLY to the injected write
        inj = torch.einsum("hrp,hrn->hpn", base["V_rank"][t], base["K_trap"][t])
        rebuilt = p["gamma_write"] + p["deferred_write"]
        assert (rebuilt - inj).abs().max().item() < 1e-5, t
        assert p["write_source_token"] == t
        assert p["gamma_write_resolution_token"] == t
        want_res = None if t == T - 1 else t + 1
        assert p["deferred_write_resolution_token"] == want_res, t
        # causal state_read norm matches the recorded online norm
        assert torch.allclose(p["state_read"].reshape(H, -1).norm(dim=-1),
                              base["online_state_norms"][t], atol=1e-5)
        # future-aware accumulator matches the recorded prefill norm
        assert torch.allclose(p["prefill_accum"].reshape(H, -1).norm(dim=-1),
                              base["prefill_accum_norms"][t], atol=1e-5)
    # final position: deferred half is zero by construction
    assert float(payloads[T - 1]["deferred_write"].abs().max()) == 0.0
    # causal recomposition: state_read[t+1] == alpha[t+1] * prefill_accum[t]
    for t in range(T - 1):
        want = base["alpha"][t + 1].view(H, 1, 1) * payloads[t]["prefill_accum"]
        assert (payloads[t + 1]["state_read"] - want).abs().max().item() < 1e-5
    print("  observer: detached clones, hostile mutation inert, write halves "
          "and resolution tokens exact, causal recomposition holds   OK")


def test_observer_rejects_bad_type():
    spec, params, d_model = _toy(seed=43)
    try:
        reference_block_forward(torch.randn(4, d_model), params, spec,
                                is_mimo=True, observer=object())
        raise AssertionError("non-callable observer should have raised")
    except TypeError:
        pass
    print("  observer: non-callable observer rejected              OK")


if __name__ == "__main__":
    print("reference recurrence self-tests (candidate oracle):")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nAll self-tests pass. These check INTERNAL invariants only.")
    print("Kernel equivalence is UNVALIDATED and remains blocked on GPU tensor")
    print("parity (gpu_probe g2/g3). Do not cite this as agreeing with the kernel.")
