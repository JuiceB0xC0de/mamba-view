"""Stage 1B: elementwise MIXER parity harness, official kernel vs candidate oracle.

WHAT CHANGED AND WHY
  The previous version of this file swept `RefSwitches` combinations and scored
  them by top-k token overlap against a recorded prediction. That is not a
  parity test. Token agreement is a coarse, order-destroying summary of a
  128k-way distribution; two materially different recurrences can produce the
  same argmax. `RefSwitches` no longer exists (Stage 1A derived every rule from
  pinned source), and readable output is no longer permitted to establish
  anything.

WHAT THIS DOES INSTEAD
  Compares the COMPLETE mixer output tensor, elementwise, from an IDENTICAL
  block input:

    1. forward_pre_hook on blk.mixer   -> the normalized tensor entering the mixer
    2. forward_hook     on blk.mixer   -> the official mixer output, BEFORE the
                                          surrounding residual addition
    3. that layer's exact weights and ssm_cfg are extracted
    4. input and weights are promoted to FP32 and fed to the candidate oracle
    5. every position and channel is compared

  The residual, the MLP, the final norm and the unembedding are all OUTSIDE the
  comparison on purpose: they would dilute a mixer discrepancy.

VERDICTS -- three, and they are not interchangeable
  PASS         elementwise mixer parity within the declared tolerances
  STOP         execution succeeded, numerical parity FAILED
  UNAVAILABLE  the official kernel cannot run here (no CUDA, no mamba_ssm, JIT
               failure). **UNAVAILABLE IS NOT PASS.** It carries no evidence
               about parity in either direction.

TOLERANCE RATIONALE (derived a priori; run `--self-check` to reproduce)
  The kernel runs bf16; the oracle runs fp32. bf16 has 8 mantissa bits, so
  eps ~= 2^-8 ~= 3.9e-3.

  ERROR MODEL. For a sum of n random-sign terms the RESULT grows as sqrt(n) too,
  so relative error stays ~eps. sqrt(n) amplification appears only under
  CANCELLATION. What actually compounds is the number of sequential rounding
  STAGES the value passes through -- in_proj, norm, the state contraction, the
  gate, out_proj -- roughly 5, giving ~sqrt(5) * eps ~= 8.7e-3 typical.

  THREE VERDICT-BEARING METRICS. Cosine and vendor_rel_p95 are both insensitive to
  SPARSE corruption: a percentile steps over a handful of bad elements (a single
  corrupted scalar measures rel_p99 = 0.000 exactly), and cosine dilutes as
  1/sqrt(N), so on a realistic 27k-element mixer output one wrong element is
  invisible to both. norm_max_err = max_abs_err / RMS(official) is therefore
  also verdict-bearing.

  Measured on synthetic models (--self-check), 9x32 tensor:
      case                        norm_max   rel_p99    cosine
      true bf16 round-trip          0.0062   3.2e-03   0.999999
      per-element eps               0.0295   1.3e-02   0.999989
      compounded over 5 stages      0.0377   2.2e-02   0.999961
      INFO sqrt(d_state) cancel     0.3325   1.2e-01   0.998821
      STRUCT one scalar             1.0693   0.0e+00   0.998013
      STRUCT one channel            0.4277   9.1e-01   0.997158
      STRUCT sparse <1%             0.8554   0.0e+00   0.997556

  Defaults: cos >= 0.9995, vendor_rel_p95 <= 1e-1, norm_max <= 0.30. The
  relative statistic exactly matches the released SISO vendor test: p95 over
  elements whose reference magnitude is at least 1e-2, with a 1e-6
  denominator epsilon. rel_p99 with the RMS-scaled floor is still reported as
  a stricter diagnostic but no longer pretends to be a supported kernel bar.
  The norm_max bar
  sits ~4x above the worst precision case and ~3x below the mildest structural
  one. All three are fixed BEFORE any GPU run. A failing run is STOP; it is not
  licence to widen them afterwards.

  The sqrt(d_state) cancellation case is INFORMATIONAL and deliberately NOT
  required to pass. If a real run lands in that band, investigate cancellation
  in the contraction -- do not widen the tolerance to accommodate it.

RELATIVE ERROR DENOMINATOR (documented, per the stage requirement)
    rel = |cand - off| / max(|off|, floor),  floor = rel_floor_frac * RMS(off)
  A pure |off| denominator explodes wherever the official output is near zero
  and would make the statistic meaningless; the floor is a fixed fraction of the
  tensor's own RMS so it scales with the layer rather than being an absolute
  magic number. Default rel_floor_frac = 1e-2. Both are printed with the result.

SUPPLEMENTAL ONLY
  Full-model logits and top-k are diagnostics. BOS is prepended explicitly.
  Top-k is reported as an ORDERED list and never converted to a set. It cannot
  produce PASS and cannot produce STOP.

COVERAGE IS PART OF THE VERDICT. Overall PASS requires that the ENTIRE expected
model x layer x sequence-length matrix executed and passed. A missing snapshot,
an out-of-range layer, a missing hook capture, or zero cases is STOP, never a
silently smaller matrix that happens to pass.

CHECKPOINT RESOLUTION is delegated to mamba3_core.resolve_checkpoint, LOCAL-ONLY.
Each model is resolved once and the same ResolvedCheckpoint serves the whole
matrix, so the official kernel (loaded from ck.path) and the candidate oracle
(loaded from ck.load_state_dict()) read the SAME snapshot -- not a mutable hub
id that could move between the two loads. ck.provenance() goes into the report,
so a run identifies its exact commit even when --revision is omitted.

This file performs NO pod run. CPU invocation reports UNAVAILABLE and prints the
GPU command. That command is UNEXECUTED and provides no parity evidence; it is
no longer "provisional" in the cache-path sense, since the hardcoded path is gone.
"""

import argparse
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from mamba3_core import (  # noqa: E402
    CheckpointResolveError, InProjSpec, ShapeContractError, resolve_checkpoint,
    rms_norm,
)
from reference_recurrence import reference_block_forward  # noqa: E402

TOKENIZER_ID = "NousResearch/Meta-Llama-3.1-8B"
TOKENIZER_REV = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"

PASS, STOP, UNAVAILABLE = "PASS", "STOP", "UNAVAILABLE"

BF16_EPS = 2.0 ** -8                 # 8 mantissa bits
D_STATE_TYPICAL = 128                # contraction depth in the released configs
N_ROUNDING_STAGES = 5                # in_proj, norm, contraction, gate, out_proj
DEFAULT_COS_TOL = 0.9995             # compounded-stages case measures 0.999961
DEFAULT_REL_TOL = 1e-1               # vendor SISO test_mamba3_siso_combined_batched
VENDOR_REL_REF_FLOOR = 1e-2          # vendor relative_error(ref_mag_mask=1e-2)
VENDOR_REL_EPS = 1e-6                # vendor relative_error denominator epsilon
DEFAULT_NORM_MAX_TOL = 0.30          # catches a >=30%-of-RMS sparse corruption;
                                     # mildest structural self-check is 0.428
DEFAULT_REL_FLOOR_FRAC = 1e-2


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------


def check_availability():
    """Return (ok, reason). UNAVAILABLE is never PASS."""
    try:
        import mamba_ssm  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"mamba_ssm not importable: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "no CUDA device: the official MIMO kernel is TileLang/CUDA only"
    try:
        import tilelang  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"tilelang not importable: {type(e).__name__}: {e}"
    return True, "ok"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def parity_metrics(cand: torch.Tensor, off: torch.Tensor,
                   rel_floor_frac: float = DEFAULT_REL_FLOOR_FRAC) -> dict:
    """Elementwise comparison over ALL positions and channels.

    cand, off: (T, C) float32, same shape. No masking, no padding.
    """
    assert cand.shape == off.shape, (cand.shape, off.shape)
    d = (cand - off).abs()

    rms = off.pow(2).mean().sqrt().item()
    floor = rel_floor_frac * rms
    rel = d / off.abs().clamp_min(floor)

    # Exact statistic used by the released SISO kernel test.  cand is the
    # source-derived/reference tensor and off is the deployed kernel tensor.
    # The mask applies ONLY to this relative percentile; absolute, cosine,
    # norm-max and every per-position metric still cover the complete tensor.
    ref_abs = cand.abs()
    vendor_mask = ref_abs >= VENDOR_REL_REF_FLOOR
    if vendor_mask.any():
        vendor_vals = (d[vendor_mask]
                       / (ref_abs[vendor_mask] + VENDOR_REL_EPS)).flatten()
        vendor_k = max(1, min(vendor_vals.numel(),
                              int(math.ceil(0.95 * vendor_vals.numel()))))
        vendor_rel_p95 = vendor_vals.kthvalue(vendor_k).values.item()
        vendor_rel_n = int(vendor_vals.numel())
    else:
        vendor_rel_p95 = 0.0
        vendor_rel_n = 0

    per_pos = d.max(dim=-1).values                      # (T,)
    T = per_pos.shape[0]
    interior = per_pos[1:-1] if T > 2 else per_pos[:0]

    cos = F.cosine_similarity(cand.reshape(1, -1), off.reshape(1, -1)).item()

    return {
        "shape": list(cand.shape),
        "n_elements": int(cand.numel()),
        "max_abs_err": d.max().item(),
        "mean_abs_err": d.mean().item(),
        # verdict-bearing. Catches SPARSE corruption that a percentile steps
        # over and that cosine dilutes as 1/sqrt(N).
        "norm_max_err": d.max().item() / max(rms, 1e-12),
        "rel_denominator": f"max(|official|, {rel_floor_frac} * RMS)",
        "rel_floor": floor,
        "official_rms": rms,
        "rel_p50": rel.median().item(),
        "rel_p99": torch.quantile(rel.flatten().float(), 0.99).item(),
        "rel_max": rel.max().item(),
        "vendor_rel_p95": vendor_rel_p95,
        "vendor_rel_n": vendor_rel_n,
        "vendor_rel_definition": (
            "p95(|candidate-official|/(|candidate|+1e-6)) over "
            "|candidate|>=1e-2; exact released SISO vendor-test statistic"
        ),
        "cosine": cos,
        "per_position_max_abs": [round(v, 8) for v in per_pos.tolist()],
        "pos_first": per_pos[0].item(),
        "pos_final": per_pos[-1].item(),
        "pos_interior_max": interior.max().item() if interior.numel() else None,
        "pos_interior_mean": interior.mean().item() if interior.numel() else None,
    }


def verdict_for(m: dict, cos_tol: float, rel_tol: float,
                norm_max_tol: float = DEFAULT_NORM_MAX_TOL) -> str:
    """All three metrics must pass. Any one failing is STOP."""
    for k in ("max_abs_err", "cosine", "norm_max_err", "vendor_rel_p95"):
        if not math.isfinite(m.get(k, float("nan"))):
            return STOP
    ok = (m["cosine"] >= cos_tol
          and m["vendor_rel_p95"] <= rel_tol
          and m["norm_max_err"] <= norm_max_tol)
    return PASS if ok else STOP


# --------------------------------------------------------------------------
# one parity case
# --------------------------------------------------------------------------


def resolve_arm(spec, ssm, where=""):
    """Tensor-derived arm is authoritative; ssm_cfg.is_mimo is a CROSS-CHECK only.

    spec.is_mimo comes from B_bias.shape[1] (mimo_rank), a realized tensor
    dimension. config.ssm_cfg.is_mimo is an advisory field that can go stale.
    Selecting the execution arm from the config would let a wrong config route a
    MIMO checkpoint down the SISO path (or vice versa) while every shape check
    still passed on the tensors it did look at.

    If the field is present and disagrees, raise with BOTH values.
    If the field is absent, the tensor-derived value stands.
    """
    if ssm is not None and "is_mimo" in ssm:
        cfg_arm = bool(ssm["is_mimo"])
        if cfg_arm != spec.is_mimo:
            raise ShapeContractError(
                f"arm disagreement{(' at ' + where) if where else ''}: "
                f"config ssm_cfg.is_mimo={cfg_arm} but tensor-derived "
                f"mimo_rank={spec.mimo_rank} implies is_mimo={spec.is_mimo}. "
                f"The tensors are authoritative; the config is stale or the "
                f"checkpoint is mislabelled.")
    return spec.is_mimo


def extract_layer_params(model, li):
    """Exact weights of one mixer, promoted to FP32 on CPU."""
    mixer = model.backbone.layers[li].mixer
    out = {}
    for name, p in mixer.named_parameters(recurse=True):
        out[name] = p.detach().float().cpu()
    for name, b in mixer.named_buffers(recurse=True):
        out[name] = b.detach().float().cpu()
    return out


def run_parity_case(model, ids, li, cfg, cos_tol, rel_tol, rel_floor_frac,
                    norm_max_tol=DEFAULT_NORM_MAX_TOL):
    """Compare official mixer output against the oracle from the same input."""
    blk = model.backbone.layers[li]
    grabbed = {}
    h_pre = blk.mixer.register_forward_pre_hook(
        lambda m, inp: grabbed.__setitem__("mixer_in", inp[0].detach()))
    h_post = blk.mixer.register_forward_hook(
        lambda m, i, o: grabbed.__setitem__("mixer_out", o.detach()))
    try:
        with torch.inference_mode():
            model(ids.unsqueeze(0))
    finally:
        h_pre.remove()
        h_post.remove()

    # A missing capture is STOP, explicitly. Silently comparing nothing, or
    # comparing a stale tensor from a previous case, would be a false PASS.
    missing = [k for k in ("mixer_in", "mixer_out") if k not in grabbed]
    if missing:
        raise RuntimeError(
            f"hook capture missing {missing} for layer {li}: the mixer did not "
            "run, or the module layout changed. Refusing to compare."
        )

    u = grabbed["mixer_in"][0].float().cpu()        # (T, d_model), normalized
    off = grabbed["mixer_out"][0].float().cpu()     # (T, d_model), pre-residual
    if u.shape[0] != ids.shape[0] or off.shape[0] != ids.shape[0]:
        raise RuntimeError(
            f"captured length {u.shape[0]}/{off.shape[0]} != input {ids.shape[0]}"
        )

    params = extract_layer_params(model, li)
    ssm = cfg["ssm_cfg"]

    # LIVE-MIXER spec: realized module shapes, not config arithmetic.
    spec = InProjSpec.from_mixer(blk.mixer)
    spec.assert_against(params["in_proj.weight"].shape[0])

    r = reference_block_forward(
        u, params, spec, resolve_arm(spec, ssm, f"L{li} mixer parity"),
        float(ssm.get("A_floor", 1e-4))
    )
    m = parity_metrics(r["out"], off, rel_floor_frac)
    m["verdict"] = verdict_for(m, cos_tol, rel_tol, norm_max_tol)
    m["layer"] = li
    m["seqlen"] = int(ids.shape[0])
    return m


# --------------------------------------------------------------------------
# supplemental full-model diagnostic (never verdict-bearing)
# --------------------------------------------------------------------------


def swiglu(x, fc1_w, fc2_w):
    """mlp.py L29-34 verbatim: y, gate = fc1(x).chunk(2); y * silu(gate).

    The SECOND half is the gate. Source-derived, not inferred from output
    readability. mlp.py L24 also rounds hidden width up to a multiple of 128,
    which is why config d_intermediate 1264 is realized as 1280.
    """
    y = x @ fc1_w.T
    y, gate = y.chunk(2, dim=-1)
    return (y * F.silu(gate)) @ fc2_w.T


def run_model(sd, cfg, ids):
    """Whole-model FP32 forward with the candidate oracle. SUPPLEMENTAL ONLY."""
    ssm = cfg["ssm_cfg"]
    # STATE-DICT spec, derived from realized tensors in the checkpoint.
    spec = InProjSpec.from_state_dict(sd, cfg, layer=0)
    arm = resolve_arm(spec, ssm, "supplemental whole-model")
    x = sd["backbone.embedding.weight"].float()[ids]
    for li in range(cfg["n_layer"]):
        p, mx = f"backbone.layers.{li}.", f"backbone.layers.{li}.mixer."
        params = {k.split("mixer.")[-1]: sd[k].float() for k in sd if k.startswith(mx)}
        h = rms_norm(x, sd[p + "norm.weight"].float())
        x = x + reference_block_forward(
            h, params, spec, arm,
            float(ssm.get("A_floor", 1e-4)))["out"]
        h2 = rms_norm(x, sd[p + "norm2.weight"].float())
        x = x + swiglu(h2, sd[p + "mlp.fc1.weight"].float(),
                       sd[p + "mlp.fc2.weight"].float())
    x = rms_norm(x, sd["backbone.norm_f.weight"].float())
    return x @ sd["lm_head.weight"].float().T


def supplemental_topk(model, sd, cfg, tok, prompt, k=5):
    """Ordered top-k from both paths. DIAGNOSTIC. Never sets a verdict."""
    ids = [tok.bos_token_id] + tok(prompt, add_special_tokens=False).input_ids
    t = torch.tensor(ids)
    with torch.inference_mode():
        off = model(t.unsqueeze(0).to("cuda")).logits[0, -1].float().cpu()
    cand = run_model(sd, cfg, t)[-1]

    def ordered(v):
        top = torch.topk(v, k)
        return [{"rank": i, "token": tok.decode([int(idx)]), "logit": round(float(val), 4)}
                for i, (idx, val) in enumerate(zip(top.indices, top.values))]

    o, c = ordered(off), ordered(cand)
    return {
        "note": ("SUPPLEMENTAL DIAGNOSTIC ONLY. Ordered top-k, never a set. "
                 "Token agreement does NOT establish recurrence parity."),
        "bos_prepended": True, "bos_id": tok.bos_token_id,
        "official_ordered": o, "candidate_ordered": c,
        "rank_order_identical": [d["token"] for d in o] == [d["token"] for d in c],
    }


# --------------------------------------------------------------------------


def case_lengths(chunk_size, is_mimo=False):
    """Arm-specific default parity surface.

    SISO covers 1, 2, a short irregular length, and two chunk-boundary lengths.
    MIMO begins at 2 because the released TileLang prefill kernel performs an
    illegal CUDA access at length 1 on H100.  The frozen atlas token contract
    has minimum valid_len=2 (BOS plus at least one content token), so the
    crashing length is outside the capture execution surface.  An explicit
    ``--lengths 1`` still runs it and reproduces the failure; it is not hidden.

    Boundary crossing matters because the kernel processes in chunks of
    chunk_size (16 for MIMO, 64 for SISO in the released configs) and the
    trapezoid reads Delta_{t+1}, so a chunk edge is where an off-by-one in the
    shifted term would first appear.
    """
    prefix = [2, 7] if is_mimo else [1, 2, 7]
    return prefix + [chunk_size + 1, 2 * chunk_size + 3]


def _corrupt_scalar(t, mag=1.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = t.clone().reshape(-1)
    out[int(torch.randint(0, out.numel(), (1,), generator=g))] += mag
    return out.reshape(t.shape)


def _corrupt_channel(t, mag=0.4):
    out = t.clone()
    out[:, out.shape[1] // 3] += mag
    return out


def _corrupt_sparse(t, frac=0.008, mag=0.8, seed=2):
    g = torch.Generator().manual_seed(seed)
    out = t.clone().reshape(-1)
    n = max(1, int(frac * out.numel()))
    idx = torch.randperm(out.numel(), generator=g)[:n]
    out[idx] += mag
    return out.reshape(t.shape)


def self_check(cos_tol, rel_tol, rel_floor_frac, norm_max_tol=DEFAULT_NORM_MAX_TOL):
    """CPU-safe calibration of the harness itself. No model, no GPU.

    Establishes that the declared tolerances separate PRECISION noise from a
    STRUCTURAL error. If a future change makes the structural case PASS, the
    harness has stopped being able to detect the thing it exists to detect.
    """
    torch.manual_seed(0)
    off = torch.randn(9, 32)
    # ERROR MODEL, corrected. For a sum of n random-sign terms the RESULT also
    # grows as sqrt(n), so relative error stays ~eps; the sqrt(n) amplification
    # appears only under CANCELLATION. What compounds is the number of
    # sequential rounding stages (in_proj, norm, contraction, gate, out_proj),
    # about 5, giving ~sqrt(5)*eps.
    stages = 5
    cases = [
        ("true bf16 round-trip", off.bfloat16().float(), PASS),
        ("per-element bf16 eps", off * (1 + torch.randn_like(off) * BF16_EPS), PASS),
        (f"compounded over {stages} rounding stages",
         off * (1 + torch.randn_like(off) * math.sqrt(stages) * BF16_EPS), PASS),
        # INFO: deliberately NOT required to pass. sqrt(d_state)*eps per element
        # is a worst-case CANCELLATION bound, not the expected regime. If a real
        # run lands here, investigate cancellation in the contraction -- do not
        # widen the tolerance to accommodate it.
        ("INFO worst-case cancellation sqrt(d_state)*eps",
         off * (1 + torch.randn_like(off) * math.sqrt(D_STATE_TYPICAL) * BF16_EPS), None),
        ("STRUCTURAL: one interior position wrong",
         torch.cat([off[:4], off[4:5] + 1.0, off[5:]]), STOP),
        ("STRUCTURAL: final position wrong",
         torch.cat([off[:-1], off[-1:] + 0.5]), STOP),
        ("STRUCTURAL: first position wrong",
         torch.cat([off[:1] + 0.5, off[1:]]), STOP),
        ("STRUCTURAL: one corrupted SCALAR", _corrupt_scalar(off), STOP),
        ("STRUCTURAL: one corrupted CHANNEL", _corrupt_channel(off), STOP),
        ("STRUCTURAL: sparse <1% of elements", _corrupt_sparse(off, 0.008), STOP),
    ]
    print(f"self-check  cos_tol={cos_tol}  rel_tol={rel_tol}  "
          f"norm_max_tol={norm_max_tol}  floor_frac={rel_floor_frac}")
    print(f"  {'case':44s} {'want':>5s} {'got':>5s} {'norm_max':>9s} "
          f"{'vrel_p95':>10s} {'cosine':>9s}")
    bad = 0
    for name, cand, want in cases:
        m = parity_metrics(cand, off, rel_floor_frac)
        got = verdict_for(m, cos_tol, rel_tol, norm_max_tol)
        if want is None:
            flag, want = "inf", "----"
        else:
            flag = "ok " if got == want else "BAD"
            if got != want:
                bad += 1
        print(f"  [{flag}] {name:44s} {want:>5s} {got:>5s} "
              f"{m['norm_max_err']:9.4f} {m['vendor_rel_p95']:10.3e} {m['cosine']:9.6f}")

    # Realistic-size demonstration of WHY norm_max is verdict-bearing. At mixer
    # scale (~27k elements) a single corrupted scalar is invisible to cosine and
    # to rel_p99; only norm_max sees it. If this case ever PASSes, the harness
    # has lost the ability to detect sparse corruption.
    big = torch.randn(35, 768)
    mb = parity_metrics(_corrupt_scalar(big), big, rel_floor_frac)
    got_all = verdict_for(mb, cos_tol, rel_tol, norm_max_tol)
    got_wo = PASS if (mb["cosine"] >= cos_tol
                      and mb["vendor_rel_p95"] <= rel_tol) else STOP
    ok = got_all == STOP
    if not ok:
        bad += 1
    print(f"\n  [{'ok ' if ok else 'BAD'}] realistic 35x768, ONE corrupted scalar")
    print(f"        with norm_max : {got_all}   (norm_max={mb['norm_max_err']:.3f})")
    print(f"        without it    : {got_wo}   "
          f"(cos={mb['cosine']:.8f}, vendor_rel_p95={mb['vendor_rel_p95']:.3e}) "
          f"<- the false-PASS surface")

    bad += 0 if _arm_resolution_checks() else 1
    print("\nself-check " + ("PASSED" if bad == 0 else f"FAILED ({bad} cases)"))
    return bad == 0


def _arm_resolution_checks():
    """The execution arm must come from tensors, with config as cross-check only.

    Uses real cached checkpoints (local-only, no downloads) so the tensor-derived
    arm is genuine, then perturbs only the config field.
    """
    ok = True
    print("\narm resolution (tensor-derived authoritative, config cross-check):")
    for name, want_mimo in (("siso-187m", False), ("mimo-187m", True)):
        try:
            ck = resolve_checkpoint(name, local_only=True)
            cfg = ck.load_config()
            sd = ck.load_state_dict()
            spec = InProjSpec.from_state_dict(sd, cfg, layer=0)
            del sd
        except CheckpointResolveError as e:
            print(f"  [skip] {name}: {e}")
            continue

        ssm = dict(cfg["ssm_cfg"])
        if resolve_arm(spec, ssm) != want_mimo or spec.is_mimo != want_mimo:
            print(f"  [BAD] {name}: agreeing config resolved wrongly")
            ok = False
        else:
            print(f"  [ok ] {name}: agreeing config -> is_mimo={want_mimo} "
                  f"(mimo_rank={spec.mimo_rank})")

        # config LIES about the arm -> must raise, reporting both values
        lying = dict(ssm)
        lying["is_mimo"] = not want_mimo
        try:
            resolve_arm(spec, lying, "self-check")
            print(f"  [BAD] {name}: lying config did not raise")
            ok = False
        except ShapeContractError as e:
            has_both = (f"is_mimo={not want_mimo}" in str(e)
                        and f"is_mimo={want_mimo}" in str(e))
            print(f"  [{'ok ' if has_both else 'BAD'}] {name}: lying config raises, "
                  f"both values reported={has_both}")
            ok = ok and has_both

        # field ABSENT -> tensor-derived value stands, no raise
        absent = {k: v for k, v in ssm.items() if k != "is_mimo"}
        try:
            got = resolve_arm(spec, absent, "self-check")
            print(f"  [{'ok ' if got == want_mimo else 'BAD'}] {name}: absent field "
                  f"-> tensor-derived is_mimo={got}")
            ok = ok and got == want_mimo
        except ShapeContractError as e:
            print(f"  [BAD] {name}: absent field raised: {e}")
            ok = False
    return ok


def build_plan(args):
    """Resolve the full model x layer x length matrix. CPU-safe and LOCAL-ONLY.

    Each model is resolved ONCE via mamba3_core.resolve_checkpoint and the
    ResolvedCheckpoint is retained for the whole matrix, so the official kernel
    and the candidate oracle are guaranteed to read the SAME snapshot. Nothing
    downloads: the runbook performs downloads explicitly before this command.
    """
    req_layers = [int(x) for x in args.layers.split(",")]
    plan, errors = [], []
    for name in args.models.split(","):
        try:
            ck = resolve_checkpoint(name, revision=args.revision, local_only=True)
        except CheckpointResolveError as e:
            errors.append(f"{name}: {e}")
            continue

        prov = ck.provenance()
        if not prov.get("path") or not prov.get("weights_file"):
            errors.append(f"{name}: incomplete provenance {prov}")
            continue
        if prov.get("resolved_commit") is None and not ck.from_local_dir:
            errors.append(
                f"{name}: resolved_commit is None for a hub-resolved snapshot; "
                f"the exact checkpoint cannot be identified in the manifest")
            continue

        cfg = ck.load_config()
        bad = [li for li in req_layers if li < 0 or li >= cfg["n_layer"]]
        if bad:
            errors.append(
                f"{name}: layers {bad} out of range (n_layer={cfg['n_layer']})")
            continue
        chunk = int(cfg["ssm_cfg"].get("chunk_size", 64))
        try:
            sd_for_arm = ck.load_state_dict()
            arm_spec = InProjSpec.from_state_dict(
                sd_for_arm, cfg, layer=req_layers[0])
            is_mimo = resolve_arm(
                arm_spec, cfg.get("ssm_cfg"), f"{name} parity plan")
            del sd_for_arm
        except (CheckpointResolveError, ShapeContractError, KeyError, ValueError) as e:
            errors.append(f"{name}: cannot derive parity arm: {type(e).__name__}: {e}")
            continue
        lengths = ([int(x) for x in args.lengths.split(",")] if args.lengths
                   else case_lengths(chunk, is_mimo=is_mimo))
        if not lengths:
            errors.append(f"{name}: zero sequence lengths")
            continue
        boundary = ({
            "unsupported_length": 1,
            "status": "OUTSIDE_CAPTURE_SURFACE",
            "reason": ("released MIMO TileLang prefill kernel performs an illegal "
                       "CUDA access at seqlen=1 on H100; frozen atlas token "
                       "contract minimum valid_len is 2"),
            "explicit_override_still_tests_it": True,
        } if is_mimo and not args.lengths else None)
        plan.append({"model": name, "ck": ck, "cfg": cfg,
                     "chunk": chunk, "layers": req_layers, "lengths": lengths,
                     "is_mimo": is_mimo, "input_boundary": boundary})
    expected = sum(len(p["layers"]) * len(p["lengths"]) for p in plan)
    return plan, errors, expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="siso-187m,mimo-187m")
    ap.add_argument("--layers", default="0,5")
    ap.add_argument("--revision", default=None,
                    help="exact checkpoint revision; resolution stays local-only")
    ap.add_argument("--lengths", default=None, help="override; default derives from chunk_size")
    ap.add_argument("--prompt", default="Mamba-3 is")
    ap.add_argument("--cos-tol", type=float, default=DEFAULT_COS_TOL)
    ap.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    ap.add_argument("--rel-floor-frac", type=float, default=DEFAULT_REL_FLOOR_FRAC)
    ap.add_argument("--norm-max-tol", type=float, default=DEFAULT_NORM_MAX_TOL,
                    help="max_abs_err / RMS(official); catches sparse corruption")
    ap.add_argument("--supplemental", action="store_true",
                    help="also print the ordered top-k diagnostic")
    ap.add_argument("--self-check", action="store_true",
                    help="CPU-only calibration of the harness; no model needed")
    ap.add_argument("--out", default="mixer_parity_report.json")
    args = ap.parse_args()

    if args.self_check:
        ok = self_check(args.cos_tol, args.rel_tol, args.rel_floor_frac,
                        args.norm_max_tol)
        sys.exit(0 if ok else 1)

    report = {
        "stage": "1B mixer parity",
        "tolerances": {"cosine": args.cos_tol,
                       "vendor_rel_p95": args.rel_tol,
                       "vendor_rel_ref_floor": VENDOR_REL_REF_FLOOR,
                       "vendor_rel_eps": VENDOR_REL_EPS,
                       "rel_p99": "diagnostic_only",
                       "norm_max": args.norm_max_tol,
                       "rel_floor_frac": args.rel_floor_frac},
        "cases": [], "verdicts": {},
    }

    # ---- coverage plan FIRST. It reads config and realized checkpoint shapes,
    #      so an invalid layer/arm or missing checkpoint STOPs before CUDA work.
    #      A configuration error is not an availability problem and must not be
    #      masked by UNAVAILABLE.
    plan, plan_errors, expected = build_plan(args)
    report["coverage"] = {"expected_cases": expected, "plan_errors": plan_errors}
    # Provenance is recorded at PLAN time, so even an UNAVAILABLE or STOP run
    # documents exactly which snapshots it would have used.
    report["provenance"] = {p["model"]: p["ck"].provenance() for p in plan}
    report["input_boundaries"] = {
        p["model"]: p["input_boundary"] for p in plan if p["input_boundary"]
    }
    if plan_errors or expected == 0:
        report["verdicts"]["overall"] = STOP
        for e in plan_errors:
            print(f"[{STOP}] coverage: {e}")
        if expected == 0:
            print(f"[{STOP}] coverage: zero cases planned")
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\noverall: {STOP}\nwrote {args.out}")
        return

    # ---- availability only after the plan is known to be valid ----
    ok, reason = check_availability()
    if not ok:
        report["verdicts"]["overall"] = UNAVAILABLE
        report["unavailable_reason"] = reason
        print(f"[{UNAVAILABLE}] {reason}")
        print(f"  (coverage plan is valid: {expected} cases would run)")
        print("\nUNAVAILABLE IS NOT PASS. No parity evidence was produced.")
        print("\nGPU command (UNEXECUTED -- provides no parity evidence).")
        print("Resolution is local-only; download the checkpoints first:\n")
        print("    python analysis/full_model_smoke.py \\")
        print("        --models siso-187m,mimo-187m --layers 0,5 --supplemental \\")
        print("        --out mixer_parity_report.json")
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")
        return

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REV)
    base_ids = [tok.bos_token_id] + tok(args.prompt, add_special_tokens=False).input_ids

    worst = PASS
    executed = 0
    for p in plan:
        name, ck, cfg, chunk = p["model"], p["ck"], p["cfg"], p["chunk"]
        lengths = p["lengths"]
        report.setdefault("provenance", {})[name] = ck.provenance()

        # Load the official model from the RESOLVED SNAPSHOT PATH, not the
        # mutable hub id: the kernel and the oracle must read the same bytes.
        model = MambaLMHeadModel.from_pretrained(
            ck.path, device="cuda", dtype=torch.bfloat16).eval()
        sd = ck.load_state_dict()

        # Cross-check the two specification sources. A disagreement means the
        # live module and the checkpoint on disk describe different geometry,
        # which invalidates every comparison built on either one.
        spec_mismatch = None
        try:
            for li in p["layers"]:
                a = InProjSpec.from_mixer(model.backbone.layers[li].mixer)
                b = InProjSpec.from_state_dict(sd, cfg, layer=li)
                diffs = [f for f in ("d_inner", "d_state", "ngroups", "mimo_rank",
                                     "nheads", "headdim", "n_rope_angles", "total")
                         if getattr(a, f) != getattr(b, f)]
                resolve_arm(a, cfg.get("ssm_cfg"), f"{name} L{li} live-mixer")
                resolve_arm(b, cfg.get("ssm_cfg"), f"{name} L{li} state-dict")
                if diffs:
                    spec_mismatch = (
                        f"{name} L{li}: live-mixer and state-dict specs disagree on "
                        + ", ".join(f"{f} ({getattr(a, f)} vs {getattr(b, f)})"
                                    for f in diffs))
                    break
        except ShapeContractError as e:
            spec_mismatch = f"{name}: {e}"

        if spec_mismatch:
            print(f"[{STOP}] {spec_mismatch}")
            report.setdefault("spec_mismatches", []).append(spec_mismatch)
            worst = STOP
            del sd, model
            torch.cuda.empty_cache()
            continue

        for li in p["layers"]:
            for L in lengths:
                # single sequence, NO padding: padding belongs to the token-contract stage
                ids = torch.tensor(
                    (base_ids * (L // len(base_ids) + 1))[:L], device="cuda")
                try:
                    m = run_parity_case(model, ids, li, cfg, args.cos_tol,
                                        args.rel_tol, args.rel_floor_frac,
                                        args.norm_max_tol)
                except Exception as e:  # noqa: BLE001
                    m = {"model": name, "layer": li, "seqlen": L,
                         "verdict": STOP, "error": f"{type(e).__name__}: {e}"}
                m["model"] = name
                m["chunk_size"] = chunk
                m["crosses_chunk_boundary"] = L > chunk
                report["cases"].append(m)
                executed += 1
                if m["verdict"] == STOP:
                    worst = STOP
                print(f"  {name:10s} L{li} len={L:<5d} chunk={chunk:<3d} "
                      f"{m['verdict']:5s} "
                      f"cos={m.get('cosine', float('nan')):.6f} "
                      f"vrel_p95={m.get('vendor_rel_p95', float('nan')):.4e} "
                      f"rel_p99_diag={m.get('rel_p99', float('nan')):.4e} "
                      f"norm_max={m.get('norm_max_err', float('nan')):.4f}")

        if args.supplemental:
            # A supplemental failure is a DIAGNOSTIC error. It is recorded and
            # never alters, erases, or upgrades the mixer-parity verdict.
            try:
                report.setdefault("supplemental", {})[name] = supplemental_topk(
                    model, sd, cfg, tok, args.prompt)
            except Exception as e:  # noqa: BLE001
                report.setdefault("supplemental", {})[name] = {
                    "diagnostic_error": f"{type(e).__name__}: {e}",
                    "note": ("supplemental only; the mixer-parity verdict is "
                             "unaffected by this failure"),
                }
                print(f"  [diagnostic] supplemental top-k failed for {name}: "
                      f"{type(e).__name__}: {e} (verdict unchanged)")
        del sd, model
        torch.cuda.empty_cache()

    # ---- coverage gate: incomplete execution is STOP, not a smaller PASS ----
    report["coverage"]["executed_cases"] = executed
    report["coverage"]["complete"] = executed == expected
    if executed != expected:
        print(f"[{STOP}] coverage incomplete: executed {executed} of {expected}")
        worst = STOP

    report["verdicts"]["overall"] = worst
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\noverall: {worst}")
    if worst == STOP:
        print("Numerical parity FAILED. The oracle is not validated; do not")
        print("proceed to capture. Inspect per-position errors: a first-position")
        print("failure implicates BOS/phase init, a final-position failure")
        print("implicates the trapezoid shifted term, and a chunk-boundary")
        print("failure implicates chunk bookkeeping.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
