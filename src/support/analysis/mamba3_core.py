"""B0-1: canonical Mamba-3 primitives shared by every Stage B and Stage C artifact.

Single source of truth for in_proj slicing, the recurrence derivations, B/C
reconstruction, and the rank decomposition. Duplicating any of these across
scripts is how the two arms drift apart silently, so everything imports here.

EVERY formula below is transcribed from vendored source, with the file and line
recorded next to it. If upstream changes, this module is what gets re-verified.

Source of record:
  M  = upstream/pypi_232_post1/mamba_ssm/modules/mamba3.py
  K  = upstream/tilelang/mamba3/mamba3_mimo_fwd.py
  N  = upstream/hrsvrn_full/mamba-og/mamba_ssm/ops/triton/layernorm_gated.py

Verified pipeline order (M L177-206, K L169-221, K L265-287):

    in_proj -> split [z, x, B, C, dd_dt, dd_A, trap, angles]   M L177
    B, C reshaped to (r, g, n) with g = ngroups                M L189-190
    B_norm / C_norm applied  (plain RMSNorm over d_state)      M L205-206
    --- kernel boundary ---
    per-head bias ADDED:  k_frag[cs,r,n] += k_bias[r,n]        K L214, L220
    rotary applied to K, then trap_scale multiplies K          K L265-287

CRITICAL: with ngroups == 1, B and C are SHARED across all heads. Per-head
structure is created ONLY by the additive bias, inside the kernel. A forward
hook on in_proj therefore cannot observe per-head B/C; it must be reconstructed
with reconstruct_bc() below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

IN_PROJ_ORDER = ("z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angles")

DEFAULT_ORG = "state-spaces"
MODEL_PREFIX = "mamba3-"
WEIGHT_NAMES = ("pytorch_model.bin", "model.safetensors")


class ShapeContractError(ValueError):
    """Raised when realized tensor shapes violate the in_proj contract.

    Messages always name the arm, the tensor, the expected shape and the actual
    shape, so a failure identifies the offending tensor without a debugger.
    """


class CheckpointResolveError(FileNotFoundError):
    """Raised when a checkpoint cannot be resolved to a usable directory."""

SOURCE_RECORD = {
    "module": "upstream/pypi_232_post1/mamba_ssm/modules/mamba3.py",
    "mimo_kernel": "upstream/tilelang/mamba3/mamba3_mimo_fwd.py",
    "rmsnorm": "upstream/hrsvrn_full/mamba-og/mamba_ssm/ops/triton/layernorm_gated.py",
    "split_order": IN_PROJ_ORDER,
    "bias_application": "additive, per head, inside kernel (K L214/L220)",
    "trap_sigmoid": "applied inside kernel; prefill receives a RAW logit (K L184/L195)",
    "A_activation": "A = -heavy_tail_activation(dd_A), clamp max=-A_floor (M L194-195)",
}


def assert_runtime(cfg: dict, strict: bool = True) -> dict:
    """Record and check the runtime contract. Abort rather than warn.

    Returns the manifest dict that must be attached to every run record.
    """
    import torch as _t

    manifest = {
        "torch": _t.__version__,
        "cuda_available": _t.cuda.is_available(),
        "device_name": _t.cuda.get_device_name(0) if _t.cuda.is_available() else None,
        **SOURCE_RECORD,
        **cfg,
    }
    try:
        import tilelang

        manifest["tilelang"] = getattr(tilelang, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - tilelang absent on CPU-only boxes, which is fine
        manifest["tilelang"] = None
        if strict and cfg.get("requires_kernel"):
            raise RuntimeError("kernel path requested but tilelang is not importable")
    return manifest


# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """M L27-40.  f(x) = 1 + x for x >= 0 ; 1 / (1 - x) for x < 0.

    Positive, continuous, differentiable at 0, and f(0) == 1 exactly. That last
    fact is what makes the zero-input retention prior well defined: A(0) = -1,
    leaving the A_floor clamp inactive.

    >>> float(heavy_tail_activation(torch.zeros(1)))
    1.0
    """
    neg = x.clamp_max(0.0)
    pos = x.clamp_min(0.0)
    return torch.where(x >= 0, 1.0 + pos, 1.0 / (1.0 - neg))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Plain RMSNorm over the LAST dim, matching N L415-437 called with z=None.

    B_norm / C_norm are RMSNorm(d_state), weight initialized to ones, bias None,
    group_size None. No gating is used for B and C.
    """
    dt = x.dtype
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rms * weight.float()).to(dt)


# --------------------------------------------------------------------------
# in_proj layout
# --------------------------------------------------------------------------


@dataclass
class InProjSpec:
    """Sizes and slice bounds of the concatenated in_proj output.

    d_in_proj = 2*d_inner + 2*d_state*ngroups*mimo_rank + 3*nheads + n_rope   (M L107)
    """

    d_inner: int
    d_state: int
    ngroups: int
    mimo_rank: int
    nheads: int
    headdim: int
    n_rope_angles: int
    sizes: dict = field(default_factory=dict)
    bounds: dict = field(default_factory=dict)

    def __post_init__(self):
        self._validate_dims()
        n_bc = self.d_state * self.ngroups * self.mimo_rank
        self.sizes = {
            "z": self.d_inner,
            "x": self.d_inner,
            "B": n_bc,
            "C": n_bc,
            "dd_dt": self.nheads,
            "dd_A": self.nheads,
            "trap": self.nheads,
            "angles": self.n_rope_angles,
        }
        at = 0
        for k in IN_PROJ_ORDER:
            self.bounds[k] = (at, at + self.sizes[k])
            at += self.sizes[k]
        self.total = at

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @property
    def is_mimo(self) -> bool:
        return self.mimo_rank > 1

    @property
    def arm(self) -> str:
        return "MIMO" if self.is_mimo else "SISO"

    def _validate_dims(self) -> None:
        for name, v in (("nheads", self.nheads), ("headdim", self.headdim),
                        ("d_state", self.d_state), ("d_inner", self.d_inner),
                        ("mimo_rank", self.mimo_rank), ("ngroups", self.ngroups),
                        ("n_rope_angles", self.n_rope_angles)):
            if not isinstance(v, int) or v < 1:
                raise ShapeContractError(
                    f"[{self.arm}] {name}: expected a positive int, got {v!r}")

        if self.nheads * self.headdim != self.d_inner:
            raise ShapeContractError(
                f"[{self.arm}] nheads*headdim != d_inner: "
                f"{self.nheads}*{self.headdim} = {self.nheads * self.headdim}, "
                f"expected d_inner={self.d_inner}. A headdim of 1 usually means "
                f"it was read from B_bias.shape[-2], which is mimo_rank, not headdim."
            )

        if 2 * self.n_rope_angles > self.d_state:
            raise ShapeContractError(
                f"[{self.arm}] n_rope_angles={self.n_rope_angles} needs "
                f"{2 * self.n_rope_angles} channels but d_state={self.d_state}; "
                f"rotary pairs index n against n + d_state//2"
            )

    def _validate_slices(self, in_proj_out_features: int | None = None) -> None:
        """Every slice: positive width, contiguous, non-overlapping, exact total."""
        prev_end, seen = 0, []
        for k in IN_PROJ_ORDER:
            lo, hi = self.bounds[k]
            if hi - lo != self.sizes[k] or self.sizes[k] < 1:
                raise ShapeContractError(
                    f"[{self.arm}] slice {k}: bounds ({lo},{hi}) width {hi - lo} "
                    f"!= declared size {self.sizes[k]}")
            if lo != prev_end:
                raise ShapeContractError(
                    f"[{self.arm}] slice {k} starts at {lo} but previous slice "
                    f"ended at {prev_end}: slices must be contiguous")
            for (pk, plo, phi) in seen:
                if not (hi <= plo or lo >= phi):
                    raise ShapeContractError(
                        f"[{self.arm}] slice {k} ({lo},{hi}) overlaps {pk} ({plo},{phi})")
            seen.append((k, lo, hi))
            prev_end = hi

        if prev_end != self.total:
            raise ShapeContractError(
                f"[{self.arm}] slice walk ended at {prev_end} but total={self.total}")

        if in_proj_out_features is not None:
            self.assert_against(in_proj_out_features)

    def assert_against(self, in_proj_out_features: int) -> None:
        if self.total != in_proj_out_features:
            raise ShapeContractError(
                f"[{self.arm}] in_proj width mismatch: computed {self.total} "
                f"from {self.sizes}, realized in_proj.weight.shape[0]="
                f"{in_proj_out_features} (difference {self.total - in_proj_out_features})"
            )

    # ------------------------------------------------------------------
    # construction from realized tensors
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_check(arm, name, tensor, expected, extra=""):
        actual = tuple(tensor.shape)
        if actual != tuple(expected):
            raise ShapeContractError(
                f"[{arm}] tensor {name}: expected shape {tuple(expected)}, "
                f"got {actual}{(' -- ' + extra) if extra else ''}")

    @classmethod
    def _derive(cls, get, has, ngroups_hint=None, cfg=None, arm_hint=None):
        """Shared derivation. `get(name)->tensor`, `has(name)->bool`.

        REALIZED SHAPES WIN. Config fields are used only where a realized tensor
        cannot express the quantity (ngroups), and are then VERIFIED against the
        realized in_proj width rather than trusted.
        """
        dt_bias = get("dt_bias")
        if dt_bias.ndim != 1:
            raise ShapeContractError(
                f"dt_bias: expected 1-D (nheads,), got {tuple(dt_bias.shape)}")
        nheads = int(dt_bias.shape[0])

        out_w = get("out_proj.weight")            # (d_model, d_inner)
        if out_w.ndim != 2:
            raise ShapeContractError(
                f"out_proj.weight: expected 2-D, got {tuple(out_w.shape)}")
        d_inner = int(out_w.shape[1])

        if d_inner % nheads:
            raise ShapeContractError(
                f"d_inner={d_inner} from out_proj.weight is not divisible by "
                f"nheads={nheads} from dt_bias")
        headdim = d_inner // nheads               # NEVER from B_bias.shape[-2]

        B_bias = get("B_bias")                    # (nheads, mimo_rank, d_state)
        if B_bias.ndim != 3:
            raise ShapeContractError(
                f"B_bias: expected 3-D (nheads, mimo_rank, d_state), got "
                f"{tuple(B_bias.shape)}")
        if int(B_bias.shape[0]) != nheads:
            raise ShapeContractError(
                f"B_bias.shape[0]={int(B_bias.shape[0])} != nheads={nheads}")
        mimo_rank = int(B_bias.shape[1])
        d_state = int(B_bias.shape[2])
        arm = "MIMO" if mimo_rank > 1 else "SISO"
        if arm_hint and arm_hint.upper() != arm:
            raise ShapeContractError(
                f"arm mismatch: caller said {arm_hint}, realized mimo_rank="
                f"{mimo_rank} implies {arm}")

        cls._shape_check(arm, "C_bias", get("C_bias"), (nheads, mimo_rank, d_state))
        for nm in ("B_norm.weight", "C_norm.weight"):
            cls._shape_check(arm, nm, get(nm), (d_state,),
                             "must match d_state from B_bias.shape[2]")

        # --- MIMO requires all three; SISO must resolve WITHOUT them ---
        mimo_names = ("mimo_x", "mimo_z", "mimo_o")
        present = [n for n in mimo_names if has(n)]
        if arm == "MIMO":
            missing = [n for n in mimo_names if n not in present]
            if missing:
                raise ShapeContractError(
                    f"[MIMO] mimo_rank={mimo_rank} but missing {missing}; "
                    f"MIMO requires all of {list(mimo_names)}")
            for n in mimo_names:
                cls._shape_check(arm, n, get(n), (nheads, mimo_rank, headdim),
                                 "last axis is headdim, not d_state")
        elif present:
            raise ShapeContractError(
                f"[SISO] mimo_rank=1 but found {present}; released SISO "
                f"checkpoints carry no mimo_* tensors")

        # --- ngroups: not expressible as a standalone realized shape ---
        ngroups = ngroups_hint
        if ngroups is None and cfg:
            ngroups = (cfg.get("ssm_cfg") or {}).get("ngroups")
        if ngroups is None:
            ngroups = 1
        ngroups = int(ngroups)

        # --- n_rope_angles: DERIVED as the residual of the realized width ---
        total = int(get("in_proj.weight").shape[0])
        fixed = 2 * d_inner + 2 * d_state * ngroups * mimo_rank + 3 * nheads
        n_rope = total - fixed
        if n_rope < 1:
            raise ShapeContractError(
                f"[{arm}] in_proj residual for rope angles is {n_rope} "
                f"(total={total}, z+x+B+C+dt/A/trap={fixed}); ngroups={ngroups} "
                f"is probably wrong")

        # cross-check against config, which is advisory only
        if cfg:
            rf = (cfg.get("ssm_cfg") or {}).get("rope_fraction")
            if rf:
                expect = d_state // int(2 / float(rf))
                if expect != n_rope:
                    raise ShapeContractError(
                        f"[{arm}] n_rope_angles: realized residual {n_rope} != "
                        f"{expect} implied by config rope_fraction={rf} "
                        f"(d_state//int(2/rope_fraction)); config may be stale")

        spec = cls(d_inner=d_inner, d_state=d_state, ngroups=ngroups,
                   mimo_rank=mimo_rank, nheads=nheads, headdim=headdim,
                   n_rope_angles=n_rope)
        spec._validate_slices(total)
        return spec

    @classmethod
    def from_mixer(cls, mixer, arm_hint: str | None = None) -> "InProjSpec":
        """Build from a live mixer module. Realized shapes only."""
        def get(name):
            obj = mixer
            for part in name.split("."):
                obj = getattr(obj, part)
            return obj

        def has(name):
            try:
                get(name)
                return True
            except AttributeError:
                return False

        return cls._derive(get, has,
                           ngroups_hint=getattr(mixer, "num_bc_heads", None),
                           arm_hint=arm_hint)

    @classmethod
    def from_state_dict(cls, sd, cfg=None, layer: int = 0,
                        arm_hint: str | None = None) -> "InProjSpec":
        """Build from a checkpoint state_dict without constructing a model."""
        prefix = f"backbone.layers.{layer}.mixer."

        def get(name):
            key = prefix + name
            if key not in sd:
                raise ShapeContractError(f"missing tensor {key} in state_dict")
            return sd[key]

        def has(name):
            return (prefix + name) in sd

        return cls._derive(get, has, cfg=cfg, arm_hint=arm_hint)


# --------------------------------------------------------------------------
# canonical checkpoint resolver
# --------------------------------------------------------------------------


@dataclass
class ResolvedCheckpoint:
    """One resolved checkpoint. `path` is a real snapshot directory."""

    repo_id: str | None
    path: str
    # what the CALLER asked for; None means "whatever the cache/hub resolves to"
    requested_revision: str | None
    # what was ACTUALLY resolved, recovered from the HF snapshot path. This is
    # the field a manifest should record: it pins the exact checkpoint even when
    # requested_revision is None. None only when the layout cannot express it.
    resolved_commit: str | None
    config_path: str
    weights_path: str
    from_local_dir: bool

    def provenance(self) -> dict:
        """Manifest-ready identity of this checkpoint."""
        return {
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "resolved_commit": self.resolved_commit,
            "path": self.path,
            "weights_file": self.weights_path.rsplit("/", 1)[-1],
            "from_local_dir": self.from_local_dir,
            "sharded": False,  # resolver refuses sharded checkpoints outright
        }

    def load_config(self) -> dict:
        import json

        with open(self.config_path) as fh:
            return json.load(fh)

    def load_state_dict(self, map_location="cpu"):
        if self.weights_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(self.weights_path, device=map_location)
        return torch.load(self.weights_path, map_location=map_location,
                          weights_only=True)


def normalize_model_id(model: str) -> str:
    """Accept a full id, a short alias, or a bare repo name.

      state-spaces/mamba3-siso-187m -> unchanged
      mamba3-siso-187m              -> state-spaces/mamba3-siso-187m
      siso-187m                     -> state-spaces/mamba3-siso-187m
    """
    m = model.strip().rstrip("/")
    if "/" in m:
        return m
    if not m.startswith(MODEL_PREFIX):
        m = MODEL_PREFIX + m
    return f"{DEFAULT_ORG}/{m}"


def resolve_checkpoint(model: str, revision: str | None = None,
                       local_only: bool = True) -> ResolvedCheckpoint:
    """Resolve a checkpoint to a directory containing config.json + weights.

    Replaces per-script globbing of a hardcoded, user-specific cache path.
    Uses huggingface_hub.snapshot_download, so HF_HOME / HF_HUB_CACHE and the
    standard cache layout are respected by the library rather than reimplemented.

    model       full id, short alias, or an explicit snapshot DIRECTORY
    revision    exact revision, passed through when supplied
    local_only  True (default) never touches the network. Self-checks and CPU
                validation MUST use this. Set False only where a download is
                explicitly intended.
    """
    import os

    # explicit directory wins, and never triggers any hub lookup
    if os.path.isdir(model):
        return _verify_dir(model, repo_id=None, requested_revision=revision,
                           from_local_dir=True)

    repo_id = normalize_model_id(model)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise CheckpointResolveError(
            f"huggingface_hub is required to resolve {repo_id!r}; "
            f"pass an explicit snapshot directory instead ({e})") from e

    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=local_only,
            allow_patterns=["config.json", "*.safetensors", "pytorch_model.bin",
                            "*.json"],
        )
    except Exception as e:  # noqa: BLE001 - surface the cause, do not silently fall back
        hint = ""
        if local_only:
            hint = (" (local_only=True: it is not in the cache and no download "
                    "was attempted)")
        raise CheckpointResolveError(
            f"could not resolve {repo_id!r} revision={revision!r}{hint}: "
            f"{type(e).__name__}: {e}") from e

    return _verify_dir(path, repo_id=repo_id, requested_revision=revision,
                       from_local_dir=False)


def infer_snapshot_commit(path) -> str | None:
    """Recover the resolved commit from the HF cache layout, if present.

    Layout is  <cache>/models--<org>--<name>/snapshots/<commit>/ , so the
    directory name IS the resolved commit. Returns None for any other layout
    (e.g. a hand-assembled directory), rather than guessing.

    This is what lets a manifest identify the exact checkpoint even when the
    caller passed revision=None.
    """
    import os
    import re

    p = os.path.normpath(str(path))
    parent, name = os.path.split(p)
    if os.path.basename(parent) != "snapshots":
        return None
    return name if re.fullmatch(r"[0-9a-f]{7,64}", name) else None


def _verify_dir(path, repo_id, requested_revision, from_local_dir) -> ResolvedCheckpoint:
    """A resolved path is usable only with config.json and exactly ONE weights file.

    SHARDED CHECKPOINTS ARE REFUSED, deliberately. An earlier version selected
    sorted(shards)[0], which silently returns a PARTIAL state dict: tensors in
    the other shards are simply absent, and downstream code would raise a
    confusing missing-key error far from the cause -- or worse, a shape contract
    would pass on the subset that happened to be present.

    Sharded loading is NOT implemented and is therefore NOT claimed. Implementing
    it would require reading model.safetensors.index.json, verifying every
    declared shard exists, rejecting duplicate tensor keys across shards, and
    merging. None of the eight released Mamba-3 checkpoints are sharded, so this
    is YAGNI until a sharded checkpoint actually appears.
    """
    import os

    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        raise CheckpointResolveError(f"{path}: no config.json")

    entries = sorted(os.listdir(path))

    # index files are the unambiguous signal of a sharded checkpoint
    index_files = [f for f in entries if f.endswith(".index.json")]
    if index_files:
        raise CheckpointResolveError(
            f"{path}: found shard index {index_files}, which means a SHARDED "
            f"checkpoint. Sharded loading is not implemented, so this resolver "
            f"refuses rather than returning one shard as if it were complete. "
            f"Pass an explicit single-file checkpoint, or implement index-aware "
            f"merging (verify every declared shard, reject duplicate tensor keys)."
        )

    safet = [f for f in entries if f.endswith(".safetensors")]
    bins = [f for f in entries if f.endswith(".bin")]
    if len(safet) > 1 or len(bins) > 1:
        raise CheckpointResolveError(
            f"{path}: multiple weight files "
            f"(safetensors={safet}, bin={bins}). This is a sharded or ambiguous "
            f"checkpoint; sharded loading is not implemented. Refusing to pick "
            f"one file, which would yield a silently PARTIAL state dict."
        )

    weights = None
    for nm in WEIGHT_NAMES:
        cand = os.path.join(path, nm)
        if os.path.isfile(cand):
            weights = cand
            break
    if weights is None and len(safet) == 1:
        weights = os.path.join(path, safet[0])
    if weights is None and len(bins) == 1:
        weights = os.path.join(path, bins[0])
    if weights is None:
        raise CheckpointResolveError(
            f"{path}: no weights found (looked for {list(WEIGHT_NAMES)} "
            f"and a single *.safetensors / *.bin); contents: {entries[:10]}")

    return ResolvedCheckpoint(
        repo_id=repo_id, path=path,
        requested_revision=requested_revision,
        resolved_commit=infer_snapshot_commit(path),
        config_path=cfg, weights_path=weights, from_local_dir=from_local_dir)


def split_in_proj(out: torch.Tensor, spec: InProjSpec) -> dict:
    """Split in_proj output into named parts, with B/C reshaped as (.., r, g, n).

    out: (..., d_in_proj). Returns tensors keeping the leading dims.
    Mirrors M L177-190.
    """
    parts = {}
    for k in IN_PROJ_ORDER:
        lo, hi = spec.bounds[k]
        parts[k] = out[..., lo:hi]
    lead = out.shape[:-1]
    for k in ("B", "C"):
        parts[k] = parts[k].reshape(*lead, spec.mimo_rank, spec.ngroups, spec.d_state)
    return parts


# --------------------------------------------------------------------------
# recurrence quantities  (contract 3.1)
# --------------------------------------------------------------------------


def recurrence_quantities(
    dd_dt: torch.Tensor,
    dd_A: torch.Tensor,
    trap: torch.Tensor,
    dt_bias: torch.Tensor,
    A_floor: float = 1e-4,
) -> dict:
    """Derive the quantities the kernel actually uses. All shapes (..., nheads).

    From M L194-198 and K L182-198:

        Delta_t       = softplus(dd_dt + dt_bias)
        A_t           = -heavy_tail_activation(dd_A), clamped <= -A_floor
        ADT_t         = A_t * Delta_t              log-retention per step
        alpha_t       = exp(ADT_t)                 per-step retention factor
        lambda_t      = sigmoid(trap_t)            RAW logit in, sigmoid here
        gamma_t       = Delta_t * lambda_t
        shifted_t     = Delta_{t+1} * (1 - lambda_{t+1}),  0 at the last position
        trap_scale_t  = gamma_t + shifted_t        the scalar applied to K

    NOTE trap_scale blends token t with token t+1: it is the trapezoid rule
    integrating across a two-token window, not a per-token gain.

    The time axis is assumed to be dim -2 (i.e. (..., seqlen, nheads)).
    """
    dt = F.softplus(dd_dt.float() + dt_bias.float())
    A = -heavy_tail_activation(dd_A.float())
    A = A.clamp(max=-A_floor)
    adt = A * dt
    lam = torch.sigmoid(trap.float())

    gamma = dt * lam

    # shifted term: Delta_{t+1} (1 - lambda_{t+1}), zero at the final position
    shifted = torch.zeros_like(gamma)
    shifted[..., :-1, :] = dt[..., 1:, :] * (1.0 - lam[..., 1:, :])

    return {
        "Delta": dt,
        "A": A,
        "ADT": adt,
        "alpha": torch.exp(adt),
        "lambda": lam,
        "gamma": gamma,
        "shifted_gamma": shifted,
        "trap_scale": gamma + shifted,
        "local_halflife": math.log(2.0) / (-adt).clamp_min(1e-12),
    }


def beta_recurrence(alpha: torch.Tensor, delta: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """beta_{t+1} = alpha_{t+1} * Delta_{t+1} * (1 - lambda_{t+1}).

    Reconstructed ONLY for discussing the recurrence. The kernel does not apply
    this product; it keeps alpha in the decay/segsum path and folds the rest
    into trap_scale. Do not capture this as if it were a kernel quantity.
    """
    return alpha * delta * (1.0 - lam)


def cumulative_retention_horizon(adt: torch.Tensor, thresh: float = math.log(2.0)) -> torch.Tensor:
    """How many future positions until cumulative -sum(A*Delta) crosses ln 2.

    The faithful sequence quantity: because A and Delta change every token, a
    per-token half-life is only a local linearization. Here we walk forward and
    count positions until the accumulated log-retention crosses the threshold.

    adt: (seqlen, nheads), negative. Returns (seqlen, nheads) float, nan where
    the threshold is never crossed inside the captured window (right-censored).
    """
    seqlen, nheads = adt.shape
    decay = (-adt).clamp_min(0.0)
    csum = torch.cumsum(decay, dim=0)
    pad = torch.cat([torch.zeros(1, nheads, device=adt.device, dtype=csum.dtype), csum], 0)

    out = torch.full((seqlen, nheads), float("nan"), device=adt.device)
    for h in range(nheads):
        col = pad[:, h].contiguous()
        target = col[:-1] + thresh
        idx = torch.searchsorted(col, target.contiguous(), right=False)
        horizon = (idx - torch.arange(seqlen, device=adt.device)).float()
        horizon[idx > seqlen] = float("nan")  # censored
        out[:, h] = horizon
    return out


# --------------------------------------------------------------------------
# B / C reconstruction  (contract 3.2)
# --------------------------------------------------------------------------


def reconstruct_bc(
    raw: torch.Tensor,
    norm_weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    expand_heads: bool = True,
) -> torch.Tensor:
    """Rebuild the per-head effective B (or C) the kernel actually consumes.

    raw:         (..., rank, ngroups, d_state)   straight from split_in_proj
    norm_weight: (d_state,)                      B_norm.weight / C_norm.weight
    bias:        (nheads, rank, d_state)         B_bias / C_bias

    Returns (..., nheads, rank, d_state).

    Two steps, in this order and no other:
      1. RMSNorm over d_state                    M L205-206, before the kernel
      2. ADD the per-head bias                   K L214/L220, inside the kernel

    With ngroups == 1 the normalized tensor is shared across heads, so step 2
    is the ONLY source of per-head structure. That is precisely why hooking
    in_proj and expanding heads produces identical copies and measures nothing.

    Rotary and trap_scale are applied AFTER this point (K L265-287) and are
    handled separately by the injection-weighted variant.
    """
    normed = rms_norm(raw, norm_weight, eps=eps)  # (..., r, g, n)
    if not expand_heads:
        return normed
    nheads = bias.shape[0]
    ngroups = normed.shape[-2]
    if ngroups != 1 and ngroups != nheads:
        raise NotImplementedError(
            f"ngroups={ngroups} with nheads={nheads} is neither shared nor per-head"
        )
    # (..., r, g, n) -> (..., h, r, n)
    shared = normed.movedim(-2, -3)  # (..., g, r, n)
    if ngroups == 1:
        shared = shared.expand(*shared.shape[:-3], nheads, *shared.shape[-2:])
    return shared + bias


# --------------------------------------------------------------------------
# rank decomposition  (contract 3.2, matches the corrected Stage A analysis)
# --------------------------------------------------------------------------


def rank_split(x: torch.Tensor, rank_dim: int = -2) -> tuple:
    """Split into rank-COMMON (mean over rank) and rank-DIFFERENTIAL parts.

    The common part is what SISO also has, so it is the fair cross-arm object.
    The differential part sums to zero over rank by construction and is what
    MIMO adds. Returns (common, differential) with common keeping its dim.
    """
    common = x.mean(dim=rank_dim, keepdim=True)
    return common, x - common


def differential_norm_ratio(x: torch.Tensor, rank_dim: int = -2, dims=(-2, -1)) -> torch.Tensor:
    """||X - mean_r(X)|| / ||X||   -- UNSQUARED.

    This is what Stage A's `static_atlas.py` actually computed and reported
    (it called the field `diff_energy`, which was a misnomer). Every Stage A
    number quoted as "differential energy" is this quantity:
        mimo 187m/444m/894m/1.5b  ->  0.73 / 0.56 / 0.49 / 0.24
    Squaring those would give the energy fractions instead, so the published
    wording must say NORM RATIO. Kept so activation-side numbers stay directly
    comparable to the frozen Stage A artifact.
    """
    _, diff = rank_split(x, rank_dim)
    num = diff.pow(2).sum(dim=dims).sqrt()
    den = x.pow(2).sum(dim=dims).sqrt().clamp_min(1e-12)
    return num / den


def differential_energy(x: torch.Tensor, rank_dim: int = -2, dims=(-2, -1)) -> torch.Tensor:
    """||X - mean_r(X)||^2 / ||X||^2  -- SQUARED, a true energy fraction.

    This is the contract formula. It is the SQUARE of differential_norm_ratio,
    so do not compare values from the two interchangeably: a 0.53 norm ratio is
    a 0.28 energy fraction.

    The SAME denominator as the corrected static analysis, so activation-side
    utilization is directly comparable to the weight-side ~0.5 finding.

    WARNING, learned the hard way: when x is dominated by a constant offset
    (e.g. a bias initialized at ones that barely moved), this ratio tracks how
    far the tensor drifted rather than how differential it is. For weights,
    prefer differential_share_of_drift(). For activations the offset is not
    constant, so this ratio is the right object.
    """
    _, diff = rank_split(x, rank_dim)
    num = diff.pow(2).sum(dim=dims)
    den = x.pow(2).sum(dim=dims).clamp_min(1e-24)
    return num / den


def differential_share_of_drift(x: torch.Tensor, init_val: float, rank_dim: int = -2,
                                dims=(-2, -1)) -> torch.Tensor:
    """||X - mean_r(X)|| / ||X - init||.

    Of the movement away from initialization, what fraction is differential.
    This is the denominator that survived the Stage A retraction: the naive
    ||diff||/||X|| version fell with scale purely because larger models drift
    less from their constant init.
    """
    _, diff = rank_split(x, rank_dim)
    num = diff.pow(2).sum(dim=dims).sqrt()
    den = (x - init_val).pow(2).sum(dim=dims).sqrt().clamp_min(1e-12)
    return num / den


# --------------------------------------------------------------------------
# RoPE phase helpers  (contract 3.1)
# --------------------------------------------------------------------------


def phase_stats(angles: torch.Tensor, dim: int = 0) -> dict:
    """Circular statistics for the rotary angles.

    angles: radians, any shape; reduced along `dim`.
      concentration  R in [0,1]; 1 = perfectly aligned, 0 = uniform
      mean_angle     circular mean
      cumulative     running total, for winding
    """
    s = torch.sin(angles).mean(dim=dim)
    c = torch.cos(angles).mean(dim=dim)
    return {
        "concentration": torch.sqrt(s * s + c * c),
        "mean_angle": torch.atan2(s, c),
        "cumulative": torch.cumsum(angles, dim=dim),
    }


def winding_number(angles: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Total accumulated rotation in turns, along `dim`."""
    return angles.sum(dim=dim) / (2.0 * math.pi)


# --------------------------------------------------------------------------
# CPU self-check: shape contract + resolver, no downloads
# --------------------------------------------------------------------------

SELF_CHECK_MODELS = (
    "siso-187m", "siso-443m", "siso-893m", "siso-1.5b",
    "mimo-187m", "mimo-444m", "mimo-894m", "mimo-1.5b",
)


def _shard_negative_checks() -> bool:
    """A multi-shard directory must NOT resolve as a single complete checkpoint.

    Builds synthetic directories in a temp dir. No network, no real weights.
    """
    import json as _json
    import os
    import tempfile

    ok = True
    cases = [
        ("index file present",
         ["config.json", "model-00001-of-00002.safetensors",
          "model-00002-of-00002.safetensors", "model.safetensors.index.json"]),
        ("two safetensors shards, no index",
         ["config.json", "model-00001-of-00002.safetensors",
          "model-00002-of-00002.safetensors"]),
        ("two .bin shards",
         ["config.json", "pytorch_model-00001-of-00002.bin",
          "pytorch_model-00002-of-00002.bin"]),
    ]
    for desc, files in cases:
        with tempfile.TemporaryDirectory() as d:
            for f in files:
                with open(os.path.join(d, f), "w") as fh:
                    fh.write("{}" if f.endswith(".json") else "x")
            try:
                ck = resolve_checkpoint(d, local_only=True)
                print(f"  [BAD] {desc}: resolved to {ck.weights_path} "
                      f"-- this would be a PARTIAL state dict")
                ok = False
            except CheckpointResolveError as e:
                first = str(e).split(".")[0][:88]
                print(f"  [ok ] {desc}: refused -- {first}")

    # positive control: a single-file dir must still resolve
    with tempfile.TemporaryDirectory() as d:
        for f in ("config.json", "model.safetensors"):
            with open(os.path.join(d, f), "w") as fh:
                fh.write("{}" if f.endswith(".json") else "x")
        try:
            ck = resolve_checkpoint(d, local_only=True)
            print(f"  [ok ] single-file control resolves: "
                  f"{os.path.basename(ck.weights_path)}")
        except CheckpointResolveError as e:
            print(f"  [BAD] single-file control refused: {e}")
            ok = False

    # commit inference must recognise the HF layout and reject anything else
    fake = "/tmp/hub/models--org--name/snapshots/0123abcd4567/"
    got = infer_snapshot_commit(fake)
    if got != "0123abcd4567":
        print(f"  [BAD] commit inference: expected 0123abcd4567, got {got}")
        ok = False
    else:
        print(f"  [ok ] commit inferred from snapshot layout: {got}")
    if infer_snapshot_commit("/some/hand/made/dir") is not None:
        print("  [BAD] commit inference guessed on a non-HF layout")
        ok = False
    else:
        print("  [ok ] non-HF layout yields resolved_commit=None, not a guess")
    return ok


def _self_check(models=SELF_CHECK_MODELS, layer=0) -> bool:
    """Resolve each cached checkpoint LOCAL-ONLY and prove its in_proj decomposition."""
    ok_all, rows = True, []
    print(f"{'checkpoint':14s} {'arm':5s} {'nheads':>6s} {'headdim':>7s} "
          f"{'d_inner':>7s} {'d_state':>7s} {'rank':>4s} {'grp':>3s} "
          f"{'rope':>4s} {'in_proj total':>14s}")
    for m in models:
        try:
            ck = resolve_checkpoint(m, local_only=True)   # never downloads
            cfg = ck.load_config()
            sd = ck.load_state_dict()
            spec = InProjSpec.from_state_dict(sd, cfg, layer=layer)

            realized = int(sd[f"backbone.layers.{layer}.mixer.in_proj.weight"].shape[0])
            parts = " + ".join(f"{k}:{spec.sizes[k]}" for k in IN_PROJ_ORDER)
            assert spec.total == realized
            assert spec.nheads * spec.headdim == spec.d_inner
            rows.append((m, spec, parts, realized))
            print(f"{m:14s} {spec.arm:5s} {spec.nheads:6d} {spec.headdim:7d} "
                  f"{spec.d_inner:7d} {spec.d_state:7d} {spec.mimo_rank:4d} "
                  f"{spec.ngroups:3d} {spec.n_rope_angles:4d} "
                  f"{spec.total:7d}=={realized:<6d}")
            del sd
        except Exception as e:  # noqa: BLE001
            ok_all = False
            print(f"{m:14s} FAILED {type(e).__name__}: {e}")

    print("\ncomplete in_proj decompositions:")
    for m, spec, parts, realized in rows:
        print(f"  {m:14s} {parts} = {spec.total} (realized {realized})")

    # the specific bug this stage fixes: SISO headdim must not come from
    # B_bias.shape[-2], which is mimo_rank
    for m, spec, _, _ in rows:
        if not spec.is_mimo and spec.headdim == 1:
            print(f"\nREGRESSION: {m} resolved headdim=1 (mimo_rank leaked in)")
            ok_all = False

    print("\nresolved commits (requested vs resolved):")
    for m in models[:2] + models[4:5]:
        try:
            ck = resolve_checkpoint(m, local_only=True)
            print(f"  {m:14s} requested={str(ck.requested_revision):8s} "
                  f"resolved={ck.resolved_commit}")
        except CheckpointResolveError:
            pass

    print("\nsharded-checkpoint refusal (synthetic dirs, no downloads):")
    ok_all = _shard_negative_checks() and ok_all

    print("\nnegative cases:")
    for desc, fn in (
        ("headdim*nheads != d_inner",
         lambda: InProjSpec(d_inner=999, d_state=128, ngroups=1, mimo_rank=1,
                            nheads=24, headdim=64, n_rope_angles=32)),
        ("rope pairs exceed d_state",
         lambda: InProjSpec(d_inner=1536, d_state=8, ngroups=1, mimo_rank=1,
                            nheads=24, headdim=64, n_rope_angles=32)),
        ("unknown alias, local_only",
         lambda: resolve_checkpoint("definitely-not-a-model", local_only=True)),
    ):
        try:
            fn()
            print(f"  [BAD] {desc}: did not raise")
            ok_all = False
        except (ShapeContractError, CheckpointResolveError) as e:
            print(f"  [ok ] {desc}: {type(e).__name__}")

    print("\nself-check " + ("PASSED" if ok_all else "FAILED"))
    return ok_all


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--models", default=",".join(SELF_CHECK_MODELS))
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()
    if a.self_check:
        sys.exit(0 if _self_check(a.models.split(","), a.layer) else 1)
    print(__doc__)
