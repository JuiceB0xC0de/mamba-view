"""Memory Path parity authority: official Mamba-3 step cache vs reference states.

THE QUESTION THIS FILE ANSWERS
  Does the officially-cached decode state (angle_state, ssm_state, k_state,
  v_state, mutated in place by ``mixer.step``) equal, token by token, the
  explicitly-constructed causal state objects of the source-derived reference
  recurrence (analysis/reference_recurrence.py)? Output parity alone does NOT
  answer it: two recurrences can agree on every output while carrying different
  state conventions. This file is the SOLE authority for the Memory Path
  verdict; nothing downstream may re-derive or soften it.

THREE PATHS, SAME INPUTS, SAME PROCESS
  A  official fused prefill        mixer.forward(u) on the full sequence
  B  official token-by-token step  mixer.step(u_t, ...) with zero-allocated
                                   caches (allocate_inference_cache), snapshot
                                   of all four cache tensors after every token
  C  reference recurrence          reference_block_forward with the streaming
                                   observer, collecting per-token causal states

  All comparisons happen in ONE process with identical cloned inputs and
  weights. Separate-run CUDA nondeterminism (observed between Modal runs
  7qm94w41 and osepcarz) is exactly why cross-run comparisons are prohibited.

STATE-CONVENTION MAPPING (explicit, source-traced; names are NOT forced)
  * ssm_state (official, post-token t) = state_read[t] + gamma_write[t].
    Derivation: the step kernel (upstream/triton/mamba3/mamba3_siso_step.py
    L186-200) computes state <- alpha_t*state + beta_t*(v_{t-1} (x) k_{t-1})
    + gamma_t*(v_t (x) k_t) with beta_t = alpha_t*dt_t*(1-trap_t) and
    gamma_t = trap_t*dt_t. The reference injects token t's write once, as
    trap_scale_t = gamma_t + shifted_gamma_t with shifted_gamma_t =
    Delta_{t+1}(1-lambda_{t+1}); at step t+1 the whole state decays by
    alpha_{t+1}, so token t's deferred half becomes
    alpha_{t+1}*Delta_{t+1}(1-lambda_{t+1})*(v_t (x) k_t) -- identical to the
    kernel's beta term. Hence the official post-token cache equals the
    reference's causal read-state PLUS the same-token gamma half, and only
    that. Neither object is renamed to pretend it is the other.
  * angle_state (official, post-token t) = reference wrapped phase[t].
    Both are wrapped mod 2*pi, so raw elementwise comparison is discontinuous
    at the wrap point. DECLARED comparison: embed on the unit circle and
    compare (cos, sin) pairs elementwise -- exactly the quantities that enter
    the rotation (step kernel L162-176).
  * k_state (official) = reference pending_k[t]: rotated, biased, PRE-trap key
    (step: nxt_k_state = rotated B, mamba3.py L371).
  * v_state (official) = reference pending_v[t] squeezed over the singleton
    SISO rank (step: nxt_v_state = x, mamba3.py L370).
  * prefill_accum is FUTURE-AWARE and is never compared against any official
    cache tensor and never admissible for state-trajectory claims.

VERDICTS (precedence: STOP > STOP_STATE_PARITY > BOUNDARY >
          MEASURED_NEEDS_REVIEW > PASS)
  BOUNDARY               step path unavailable or coverage incomplete. State
                         claims remain prohibited (the current G3 boundary).
  STOP                   step executes but mixer OUTPUT parity fails.
  STOP_STATE_PARITY      outputs agree; a cache/reference state comparison
                         fails its predeclared threshold.
  MEASURED_NEEDS_REVIEW  every comparison executed, outputs pass, but at least
                         one state family has no defensible predeclared
                         threshold. NEVER promoted to PASS after the fact.
  PASS                   complete coverage, every comparison within its
                         predeclared threshold. Only PASS enables state-claim
                         mode downstream.

  No tolerance search. No post-run widening. No selecting among recurrence
  interpretations by whichever output looks closest. No top-k, logit-only, or
  norm-only evidence can produce PASS. Output tolerances are applied at their
  original full-sequence (T, C) grain; per-token output metrics are diagnostic
  only. State comparisons remain token-by-token over the complete coverage
  matrix.

THRESHOLDS
  Output comparisons reuse the a-priori tolerances of analysis/
  full_model_smoke.py verbatim (cos >= 0.9995, vendor_rel_p95 <= 1e-1,
  norm_max <= 0.30); they were derived for exactly this tensor family
  (bf16 mixer outputs vs fp32 reference). No independently justified
  thresholds exist yet for angle, K, V, or SSM cache tensors, whose numerical
  paths and storage dtypes differ from mixer outputs. Those families are
  deliberately measurement-only during calibration: the GPU probes record
  their complete error distributions and return MEASURED_NEEDS_REVIEW, never
  PASS.

This module is CPU-importable; only run_case/run_all touch CUDA. It never
imports analysis/state_trajectories.py (legacy, obsolete names, evidence only).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import traceback
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from full_model_smoke import (  # noqa: E402
    DEFAULT_COS_TOL, DEFAULT_NORM_MAX_TOL, DEFAULT_REL_TOL,
    extract_layer_params, parity_metrics, resolve_arm,
)
from mamba3_core import InProjSpec  # noqa: E402
from reference_recurrence import reference_block_forward  # noqa: E402

SCHEMA_VERSION = "memory-path-parity-v2"

MODEL_ID = "state-spaces/mamba3-siso-187m"
MODEL_REVISION = "6792c27c00f3bb41506db1066dcd1c51bb0f4b02"
TOKENIZER_ID = "meta-llama/Llama-3.1-8B"
IMAGE_REF = ("docker.io/juiceboxdocks/mamba3-atlas"
             ":1cde57a2d4981702add8c542ec75c7b2282b2ce3")

PARITY_LAYERS = (0, 5, 11)
TARGET_CAPTURE_MAX_TOKENS = 190
PARITY_SEQLENS = (2, 17, 65, TARGET_CAPTURE_MAX_TOKENS + 1)
BASE_SEED = 20260721
RUN_ROLE = "state_threshold_calibration"
CALIBRATION_PREDECESSOR_RUN_ID = "umv1n6he"

PASS = "PASS"
STOP = "STOP"
STOP_STATE_PARITY = "STOP_STATE_PARITY"
BOUNDARY = "BOUNDARY"
MEASURED_NEEDS_REVIEW = "MEASURED_NEEDS_REVIEW"
VERDICTS = (PASS, STOP, STOP_STATE_PARITY, BOUNDARY, MEASURED_NEEDS_REVIEW)

OUTPUT_FAMILIES = ("out_prefill_vs_step", "out_prefill_vs_ref",
                   "out_step_vs_ref")
STATE_FAMILIES = ("angle_state", "k_state", "v_state", "ssm_state")
ALL_FAMILIES = OUTPUT_FAMILIES + STATE_FAMILIES

_OUT_TOL = {"cos_tol": DEFAULT_COS_TOL, "rel_tol": DEFAULT_REL_TOL,
            "norm_max_tol": DEFAULT_NORM_MAX_TOL}

# Output thresholds are predeclared. State thresholds are deliberately absent
# during GPU calibration; setting them from the same run would be post-hoc
# tolerance search. A later contract revision may preregister state-family
# thresholds for an independent held-out-seed validation run.
THRESHOLDS = {
    "out_prefill_vs_step": dict(_OUT_TOL, rationale=(
        "full_model_smoke a-priori mixer-output tolerances, verbatim")),
    "out_prefill_vs_ref": dict(_OUT_TOL, rationale=(
        "full_model_smoke a-priori mixer-output tolerances, verbatim")),
    "out_step_vs_ref": dict(_OUT_TOL, rationale=(
        "full_model_smoke a-priori mixer-output tolerances, verbatim")),
    "angle_state": None,
    "k_state": None,
    "v_state": None,
    "ssm_state": None,
}

STATE_CONVENTION = {
    "state_read": ("reference CAUSAL state as read at token t: post-decay, "
                   "pre-injection; tokens <= t-1, all trapezoidal "
                   "coefficients resolved"),
    "prefill_accum": ("reference FUTURE-AWARE accumulator; index t depends on "
                      "token t+1 via trap_scale; PROHIBITED for "
                      "state-trajectory claims and never compared to caches"),
    "official_ssm_state_after_t": "state_read[t] + gamma_write[t]",
    "gamma_write": ("gamma_t * V_t (x) Bq_t; source_token=t, "
                    "resolution_token=t"),
    "deferred_write": ("shifted_gamma_t * V_t (x) Bq_t; source_token=t, "
                       "resolution_token=t+1; carried by the official "
                       "k_state/v_state cache and applied by the step "
                       "kernel's beta term at t+1"),
    "angle_comparison": "unit-circle (cos,sin) embedding, elementwise",
}


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes() -> dict:
    root = _HERE.parent
    files = {
        "analysis/memory_path_parity.py": _HERE / "memory_path_parity.py",
        "analysis/reference_recurrence.py": _HERE / "reference_recurrence.py",
        "analysis/mamba3_core.py": _HERE / "mamba3_core.py",
        "analysis/full_model_smoke.py": _HERE / "full_model_smoke.py",
        "training_memory_path_observer.py":
            root / "training_memory_path_observer.py",
    }
    return {k: _sha256_file(p) for k, p in files.items() if p.exists()}


# ---------------------------------------------------------------------------
# per-tensor comparison
# ---------------------------------------------------------------------------


def circle_embed(angles: torch.Tensor) -> torch.Tensor:
    """Wrapped angles -> concatenated (cos, sin), removing the 2*pi seam."""
    a = angles.float()
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


def compare_tensors(candidate: torch.Tensor, official: torch.Tensor,
                    family: str, thresholds: dict | None = None,
                    verdict_bearing: bool = True) -> dict:
    """Full-tensor comparison. Returns full_model_smoke metrics plus a
    comparison_verdict in {"ok", "mismatch", "needs_threshold",
    "diagnostic_only"}. ``verdict_bearing=False`` computes the identical
    metrics but cannot independently affect the parity verdict.

    candidate: reference-derived tensor. official: model-owned tensor.
    Shapes must already correspond (mapping happens in the caller, explicitly).
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    if candidate.shape != official.shape:
        raise ValueError(
            f"{family}: shape mismatch {tuple(candidate.shape)} vs "
            f"{tuple(official.shape)}; the state mapping must be explicit, "
            "not absorbed by broadcasting")
    cand = candidate.detach().float().reshape(1, -1).cpu()
    off = official.detach().float().reshape(1, -1).cpu()

    n_nonfinite = int((~torch.isfinite(cand)).sum() + (~torch.isfinite(off)).sum())
    if n_nonfinite:
        m = {"shape": list(candidate.shape), "n_elements": int(cand.numel()),
             "n_nonfinite": n_nonfinite, "comparison_verdict": "mismatch",
             "verdict_scope": ("verdict" if verdict_bearing
                               else "diagnostic_only"),
             "family": family, "dtype_candidate": str(candidate.dtype),
             "dtype_official": str(official.dtype)}
        return m

    if not cand.any() and not off.any():
        # both exactly zero (e.g. pre-first-token cache): exact agreement.
        # cosine is undefined at zero; declaring this exact-match rule here,
        # before any run, is not tolerance widening.
        m = {"shape": list(candidate.shape), "n_elements": int(cand.numel()),
             "max_abs_err": 0.0, "mean_abs_err": 0.0, "norm_max_err": 0.0,
             "cosine": 1.0, "vendor_rel_p95": 0.0, "both_exactly_zero": True}
    else:
        m = parity_metrics(cand, off)

    m["family"] = family
    m["dtype_candidate"] = str(candidate.dtype)
    m["dtype_official"] = str(official.dtype)
    m["n_nonfinite"] = n_nonfinite
    m["verdict_scope"] = ("verdict" if verdict_bearing
                          else "diagnostic_only")

    if not verdict_bearing:
        m["comparison_verdict"] = "diagnostic_only"
        return m

    decl = thresholds.get(family)
    if decl is None:
        m["comparison_verdict"] = "needs_threshold"
        return m
    ok = (m["cosine"] >= decl["cos_tol"]
          and m["vendor_rel_p95"] <= decl["rel_tol"]
          and m["norm_max_err"] <= decl["norm_max_tol"]
          and math.isfinite(m["max_abs_err"]))
    m["thresholds"] = {k: decl[k] for k in ("cos_tol", "rel_tol",
                                            "norm_max_tol")}
    m["comparison_verdict"] = "ok" if ok else "mismatch"
    return m


# ---------------------------------------------------------------------------
# verdict assembly (pure; CPU-testable)
# ---------------------------------------------------------------------------


def expected_coverage(layers=PARITY_LAYERS, seqlens=PARITY_SEQLENS) -> list:
    return [{"layer": li, "seqlen": T, "families": list(ALL_FAMILIES)}
            for li in layers for T in seqlens]


def build_verdict(case_results: list, step_available: bool,
                  layers=PARITY_LAYERS, seqlens=PARITY_SEQLENS) -> dict:
    """Compute the overall verdict from executed case results.

    case_results: list of dicts with keys layer, seqlen, comparisons
    (family -> list of per-token metric dicts), aggregate_comparisons
    (output family -> one full-sequence metric dict), and optional error.
    Output verdicts come only from aggregate_comparisons; per-token output
    rows are diagnostic. State verdicts remain token-by-token.
    """
    if not step_available:
        return {"verdict": BOUNDARY,
                "reason": "official mixer.step path unavailable; state "
                          "claims remain prohibited (G3 boundary stands)"}

    executed = {(c["layer"], c["seqlen"]): c for c in case_results
                if "error" not in c}
    missing_cases, missing_families = [], []
    any_output_mismatch, any_state_mismatch, any_needs_threshold = [], [], []

    for want in expected_coverage(layers, seqlens):
        key = (want["layer"], want["seqlen"])
        c = executed.get(key)
        if c is None:
            missing_cases.append(key)
            continue
        comps = c.get("comparisons", {})
        aggregate = c.get("aggregate_comparisons", {})
        for fam in ALL_FAMILIES:
            toks = comps.get(fam)
            if not toks:
                missing_families.append((key, fam))
                continue
            expected_tokens = (want["seqlen"] if fam in STATE_FAMILIES
                               or fam in OUTPUT_FAMILIES else None)
            if expected_tokens is not None and len(toks) != expected_tokens:
                missing_families.append((key, fam, "token_count",
                                         len(toks), expected_tokens))
                continue
            for tm in toks:
                v = tm.get("comparison_verdict")
                coord = (key[0], key[1], fam, tm.get("token"))
                if fam in OUTPUT_FAMILIES:
                    if v != "diagnostic_only":
                        missing_families.append(
                            (key, fam, "token_not_diagnostic", v))
                    continue
                if v == "mismatch":
                    any_state_mismatch.append(coord)
                elif v == "needs_threshold":
                    any_needs_threshold.append(coord)
                elif v != "ok":
                    missing_families.append((key, fam, "bad_verdict", v))
        for fam in OUTPUT_FAMILIES:
            m = aggregate.get(fam)
            coord = (key[0], key[1], fam, "full_sequence")
            if not isinstance(m, dict):
                missing_families.append((key, fam, "aggregate_missing"))
                continue
            v = m.get("comparison_verdict")
            if v == "mismatch":
                any_output_mismatch.append(coord)
            elif v != "ok":
                missing_families.append((key, fam, "aggregate_bad_verdict", v))

    errors = [c for c in case_results if "error" in c]
    detail = {
        "missing_cases": [list(k) for k in missing_cases],
        "missing_families": [list(map(str, k)) for k in missing_families],
        "output_mismatches": [list(map(str, k)) for k in any_output_mismatch],
        "state_mismatches": [list(map(str, k)) for k in any_state_mismatch],
        "needs_threshold": [list(map(str, k)) for k in any_needs_threshold],
        "case_errors": [{"layer": c.get("layer"), "seqlen": c.get("seqlen"),
                         "error": c["error"]} for c in errors],
    }
    if any_output_mismatch:
        return {"verdict": STOP, "detail": detail,
                "reason": "official step output disagrees with prefill and/or "
                          "reference beyond predeclared output tolerances"}
    if any_state_mismatch:
        return {"verdict": STOP_STATE_PARITY, "detail": detail,
                "reason": "outputs agree but at least one cache/reference "
                          "state comparison fails its predeclared threshold"}
    if missing_cases or missing_families or errors:
        return {"verdict": BOUNDARY, "detail": detail,
                "reason": "coverage incomplete: PASS requires every layer x "
                          "seqlen x tensor family to execute"}
    if any_needs_threshold:
        return {"verdict": MEASURED_NEEDS_REVIEW, "detail": detail,
                "reason": "all comparisons executed and outputs pass, but at "
                          "least one state family has no defensible "
                          "predeclared threshold; never promoted to PASS"}
    return {"verdict": PASS, "detail": detail,
            "reason": "complete coverage; every full-tensor comparison "
                      "within its predeclared threshold"}


def allowed_claims(verdict: str) -> dict:
    prohibited_base = [
        "any claim derived from prefill_accum as a causal state",
        "cross-run bitwise passivity claims (separate-run CUDA "
        "nondeterminism is unresolved)",
    ]
    if verdict == PASS:
        return {"allowed": [
            "the official decode cache trajectory equals the reference "
            "causal state trajectory under the documented mapping "
            "(ssm_state[t] == state_read[t] + gamma_write[t])",
            "token-by-token write -> retain -> read -> output decomposition "
            "on the reference recurrence, transferred to the deployed model",
            "state-claim mode may be enabled in the Memory Path recorder",
        ], "prohibited": prohibited_base}
    return {"allowed": [
        "reporting the measured errors themselves",
    ], "prohibited": prohibited_base + [
        "any kernel-state or h_t claim",
        "enabling state-claim mode in the Memory Path recorder",
        "inferring state parity from output parity",
    ]}


# ---------------------------------------------------------------------------
# report validation for downstream state-claim gating
# ---------------------------------------------------------------------------


def validate_report_for_state_claims(report: dict,
                                     expected_model: str = MODEL_ID,
                                     expected_revision: str = MODEL_REVISION,
                                     expected_source_hashes: dict | None = None,
                                     ) -> tuple[bool, str]:
    """(ok, reason). ok is True ONLY for a complete, matching PASS report."""
    if not isinstance(report, dict):
        return False, "report is not a dict"
    if report.get("schema_version") != SCHEMA_VERSION:
        return False, (f"schema_version {report.get('schema_version')!r} != "
                       f"{SCHEMA_VERSION!r}")
    if report.get("verdict") != PASS:
        return False, (f"verdict {report.get('verdict')!r} is not PASS; "
                       "state claims remain prohibited")
    if report.get("model_id") != expected_model:
        return False, f"model_id {report.get('model_id')!r} != {expected_model!r}"
    if report.get("model_revision") != expected_revision:
        return False, (f"model_revision {report.get('model_revision')!r} != "
                       f"{expected_revision!r}")
    if expected_source_hashes:
        got = report.get("source_hashes") or {}
        for k, v in expected_source_hashes.items():
            if got.get(k) != v:
                return False, f"source hash mismatch for {k}"
    # coverage must be complete IN THE REPORT, independent of the verdict field
    cases = report.get("cases")
    if not isinstance(cases, list):
        return False, "cases missing"
    seen = {(c.get("layer"), c.get("seqlen")) for c in cases
            if "error" not in c}
    for want in expected_coverage():
        if (want["layer"], want["seqlen"]) not in seen:
            return False, (f"coverage incomplete: layer {want['layer']} "
                           f"seqlen {want['seqlen']} missing")
    for c in cases:
        comps = c.get("comparisons", {})
        for fam in ALL_FAMILIES:
            if not comps.get(fam):
                return False, (f"coverage incomplete: family {fam} missing at "
                               f"layer {c.get('layer')} seqlen {c.get('seqlen')}")
        aggregate = c.get("aggregate_comparisons", {})
        for fam in OUTPUT_FAMILIES:
            m = aggregate.get(fam)
            if not isinstance(m, dict):
                return False, (f"coverage incomplete: aggregate {fam} missing "
                               f"at layer {c.get('layer')} "
                               f"seqlen {c.get('seqlen')}")
            if m.get("comparison_verdict") != "ok":
                return False, (f"aggregate {fam} is not ok at layer "
                               f"{c.get('layer')} seqlen {c.get('seqlen')}")
    return True, "complete PASS report for the pinned model/revision"


# ---------------------------------------------------------------------------
# GPU execution (only these functions require CUDA / mamba_ssm)
# ---------------------------------------------------------------------------


def step_availability() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    try:
        from mamba_ssm.modules.mamba3 import mamba3_step_fn  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        return False, f"mamba3 import failed: {type(e).__name__}: {e}"
    if mamba3_step_fn is None:
        return False, ("mamba3_step_fn is None: CuteDSL step kernel "
                       "unavailable (nvidia-cutlass-dsl / quack-kernels)")
    return True, "ok"


def make_case_input(layer: int, seqlen: int, d_model: int,
                    device, dtype=torch.bfloat16) -> torch.Tensor:
    """Deterministic isolated mixer input for one case. Recorded by seed."""
    seed = BASE_SEED + 1000 * layer + seqlen
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(seqlen, d_model, generator=g, dtype=torch.float32)
    return u.to(device=device, dtype=dtype)


class _StateCollector:
    """Observer for reference_block_forward: keeps per-token causal states."""

    def __init__(self):
        self.tokens = []
        self.final = None

    def on_token(self, t, payload):
        self.tokens.append(payload)

    def on_final(self, payload):
        self.final = payload


def run_case(model, layer: int, seqlen: int, cfg: dict) -> dict:
    """One (layer, seqlen) cell: paths A, B, C plus all required comparisons.

    Same process, identical cloned inputs and weights for all three paths.
    """
    from mamba_ssm.utils.generation import InferenceParams  # noqa: PLC0415

    mixer = model.backbone.layers[layer].mixer
    device = next(mixer.parameters()).device
    d_model = mixer.in_proj.weight.shape[1]
    u = make_case_input(layer, seqlen, d_model, device)  # (T, d_model) bf16

    rng_before = {"torch_cpu": torch.get_rng_state(),
                  "torch_cuda": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None)}
    try:
        with torch.inference_mode():
            # --- A: official fused prefill over the full sequence ---
            out_prefill = mixer(u.unsqueeze(0).clone())[0]        # (T, d_model)

            # --- B: official token-by-token step with zero caches ---
            angle_st, ssm_st, k_st, v_st = mixer.allocate_inference_cache(
                1, seqlen, device=device)
            for s in (angle_st, ssm_st, k_st, v_st):
                s.zero_()
            step_out, step_cache = [], []
            for t in range(seqlen):
                o, *_ = mixer.step(u[t:t + 1].clone(), angle_st, ssm_st,
                                   k_st, v_st)
                step_out.append(o[0].detach().float().cpu())
                step_cache.append({
                    "angle_state": angle_st.detach().float().cpu().clone(),
                    "ssm_state": ssm_st.detach().float().cpu().clone(),
                    "k_state": k_st.detach().float().cpu().clone(),
                    "v_state": v_st.detach().float().cpu().clone(),
                })

        # --- C: reference recurrence with the streaming observer ---
        params = extract_layer_params(model, layer)
        spec = InProjSpec.from_mixer(mixer)
        spec.assert_against(params["in_proj.weight"].shape[0])
        is_mimo = resolve_arm(spec, cfg.get("ssm_cfg"),
                              f"L{layer} memory-path parity")
        collector = _StateCollector()
        ref = reference_block_forward(
            u.detach().float().cpu(), params, spec, is_mimo,
            float((cfg.get("ssm_cfg") or {}).get("A_floor", 1e-4)),
            observer=collector)
    finally:
        torch.set_rng_state(rng_before["torch_cpu"])
        if rng_before["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng_before["torch_cuda"])

    if len(collector.tokens) != seqlen or len(step_cache) != seqlen:
        raise RuntimeError(
            f"token coverage broken: ref {len(collector.tokens)}, "
            f"step {len(step_cache)}, want {seqlen}")

    comps = {fam: [] for fam in ALL_FAMILIES}
    pre = out_prefill.detach().float().cpu()
    for t in range(seqlen):
        p = collector.tokens[t]
        sc = step_cache[t]
        if p["pending_v"].shape[1] != 1:
            raise RuntimeError(
                "official v_state is (batch, H, P): comparing it against a "
                f"rank-{p['pending_v'].shape[1]} pending_v has no declared "
                "mapping; refusing to guess")
        row = {
            "out_prefill_vs_step":
                (step_out[t], pre[t]),
            "out_prefill_vs_ref":
                (ref["out"][t], pre[t]),
            "out_step_vs_ref":
                (ref["out"][t], step_out[t]),
            # cache (1,H,K) vs ref phase (H,K); circle embedding, declared
            "angle_state":
                (circle_embed(p["phase"]), circle_embed(sc["angle_state"][0])),
            # cache (1,R,H,N) vs ref pending_k (H,R,N): explicit permute
            "k_state":
                (p["pending_k"].permute(1, 0, 2),
                 sc["k_state"][0]),
            # cache (1,H,P) vs ref pending_v (H,R=1,P): explicit squeeze
            "v_state":
                (p["pending_v"].squeeze(1), sc["v_state"][0]),
            # cache (1,H,P,N) vs the DOCUMENTED mapping, not a renamed object
            "ssm_state":
                (p["state_read"] + p["gamma_write"], sc["ssm_state"][0]),
        }
        for fam, (cand, off) in row.items():
            m = compare_tensors(
                cand, off, fam,
                verdict_bearing=(fam in STATE_FAMILIES),
            )
            m["token"] = t
            comps[fam].append(m)

    step_full = torch.stack(step_out, dim=0)
    ref_full = ref["out"].detach().float().cpu()
    aggregate_pairs = {
        "out_prefill_vs_step": (step_full, pre),
        "out_prefill_vs_ref": (ref_full, pre),
        "out_step_vs_ref": (ref_full, step_full),
    }
    aggregate_comparisons = {}
    for fam, (cand, off) in aggregate_pairs.items():
        m = compare_tensors(cand, off, fam, verdict_bearing=True)
        m["verdict_grain"] = "full_sequence"
        m["token_count"] = seqlen
        aggregate_comparisons[fam] = m

    return {"layer": layer, "seqlen": seqlen,
            "input_seed": BASE_SEED + 1000 * layer + seqlen,
            "comparisons": comps,
            "aggregate_comparisons": aggregate_comparisons}


def worst_coordinates(case_results: list) -> dict:
    """Worst per-token norm_max_err coordinate (diagnostic/state detail)."""
    worst = {}
    for c in case_results:
        for fam, toks in c.get("comparisons", {}).items():
            for tm in toks:
                e = tm.get("norm_max_err")
                if e is None or not math.isfinite(e):
                    continue
                if fam not in worst or e > worst[fam]["norm_max_err"]:
                    worst[fam] = {"layer": c["layer"], "seqlen": c["seqlen"],
                                  "token": tm.get("token"),
                                  "norm_max_err": e,
                                  "cosine": tm.get("cosine"),
                                  "max_abs_err": tm.get("max_abs_err")}
    return worst


def worst_aggregate_coordinates(case_results: list) -> dict:
    """Independent extrema for verdict-bearing full-sequence outputs.

    Each verdict metric can peak in a different layer/sequence case.  Keeping
    one coordinate chosen by norm_max_err would make the reported worst
    vendor_rel_p95 and cosine potentially false, even though the underlying
    case records are complete.
    """
    extrema = {}
    for c in case_results:
        for fam, m in c.get("aggregate_comparisons", {}).items():
            fam_extrema = extrema.setdefault(fam, {})
            coordinate = {"layer": c["layer"], "seqlen": c["seqlen"]}
            for metric, direction in (("norm_max_err", "max"),
                                      ("vendor_rel_p95", "max"),
                                      ("cosine", "min"),
                                      ("max_abs_err", "max")):
                value = m.get(metric)
                if value is None or not math.isfinite(value):
                    continue
                previous = fam_extrema.get(metric)
                is_worse = (previous is None
                            or (direction == "max"
                                and value > previous["value"])
                            or (direction == "min"
                                and value < previous["value"]))
                if is_worse:
                    fam_extrema[metric] = {
                        "value": value,
                        **coordinate,
                        "comparison_verdict": m.get("comparison_verdict"),
                    }
    return extrema


def environment_provenance() -> dict:
    prov = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        prov["gpu_name"] = torch.cuda.get_device_name(0)
        prov["compute_capability"] = ".".join(
            map(str, torch.cuda.get_device_capability(0)))
    for pkg in ("triton", "mamba_ssm"):
        try:
            mod = __import__(pkg)
            prov[pkg] = getattr(mod, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            prov[pkg] = f"unavailable: {type(e).__name__}"
    for dist in ("apache-tvm-ffi", "tilelang", "nvidia-cutlass-dsl",
                 "quack-kernels"):
        try:
            prov[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            prov[dist] = "unavailable"
    return prov


def run_all(model, cfg: dict, checkpoint_resolution: dict | None = None,
            layers=PARITY_LAYERS, seqlens=PARITY_SEQLENS) -> dict:
    """Full coverage matrix -> complete parity report (JSON-serializable)."""
    ok, reason = step_availability()
    geometry = {}
    try:
        mixer0 = model.backbone.layers[layers[0]].mixer
        geometry = {"nheads": mixer0.nheads, "headdim": mixer0.headdim,
                    "d_state": mixer0.d_state,
                    "n_rope_angles": mixer0.num_rope_angles,
                    "mimo_rank": mixer0.mimo_rank,
                    "n_layers": len(model.backbone.layers)}
    except Exception as e:  # noqa: BLE001
        geometry = {"error": f"{type(e).__name__}: {e}"}

    cases = []
    if ok:
        for li in layers:
            for T in seqlens:
                try:
                    cases.append(run_case(model, li, T, cfg))
                except Exception as e:  # noqa: BLE001
                    cases.append({"layer": li, "seqlen": T,
                                  "error": f"{type(e).__name__}: {e}",
                                  "traceback": traceback.format_exc()})

    v = build_verdict(cases, step_available=ok, layers=layers,
                      seqlens=seqlens)
    verdict = v["verdict"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "verdict_reason": v.get("reason"),
        "verdict_detail": v.get("detail"),
        "step_available": ok,
        "step_availability_reason": reason,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": TOKENIZER_ID,
        "docker_image": IMAGE_REF,
        "checkpoint_resolution": checkpoint_resolution,
        "source_hashes": source_hashes(),
        "environment": environment_provenance(),
        "geometry": geometry,
        "coverage_expected": expected_coverage(layers, seqlens),
        "coverage_layers": list(layers),
        "coverage_seqlens": list(seqlens),
        "run_role": RUN_ROLE,
        "base_seed": BASE_SEED,
        "target_capture_max_tokens": TARGET_CAPTURE_MAX_TOKENS,
        "calibration_predecessor_run_id": CALIBRATION_PREDECESSOR_RUN_ID,
        "thresholds": {k: (dict(v_) if v_ else None)
                       for k, v_ in THRESHOLDS.items()},
        "state_convention": STATE_CONVENTION,
        "worst_coordinates": worst_coordinates(cases),
        "worst_aggregate_coordinates": worst_aggregate_coordinates(cases),
        "cases": cases,
        "claims": allowed_claims(verdict),
        "state_claims_enabled": verdict == PASS,
    }
    return report


def write_report(report: dict, path) -> str:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True,
                               default=str) + "\n")
    return _sha256_file(path)
