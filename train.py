#!/opt/mamba3-venv/bin/python
"""Observed trainer for RunPod. Same run as the Modal trainer, no Modal.

This is a port, not a rewrite.  Every decision about what the run *is* was made
in ``simplified/trainer.py`` and is preserved here exactly: fixed dataset order
so a row index equals a step index, constant learning rate with no scheduler,
one row per optimizer step, cameras fired at named epoch-local steps, and
``grad_clip`` carried on the scene because Camera 3's observer runs its own
clip cross-check and would otherwise police a default of 1.0 and fail every
step where the real clip does not fire.

What changed is only the plumbing Modal used to own:

  * **No GPU snapshot.**  Modal restored a step-0 scene with warm Triton
    kernels in seconds.  There is no snapshot here, so the warmup runs at every
    start and the first launch pays the full compile.  The warmup is otherwise
    identical, forward and backward only with no optimizer in existence, so the
    weights are provably untouched rather than restored-and-hoped, and the RNG
    state is saved and restored around it.
  * **No Volumes.**  The camera locker is a directory and the output root is a
    directory.  Every ``volume.commit()`` is gone because a local write is
    already durable; that call was Modal's way of making a write visible
    outside the container and has no analogue.
  * **No remote invocation.**  ``modal.Cls.from_name`` and the deploy step are
    replaced by running this file.

Two behaviours worth knowing about because they differ from the Modal default:

  * ``--epochs`` is a flag here.  It was a module constant in the original,
    which meant changing it cost a snapshot rebuild.  That cost does not exist
    any more, so it stops being a constant.
  * Per-observation HuggingFace upload stays disabled.  Each point was a
    separate commit into one growing dataset repo, so the cost climbed as the
    run went, from ~6s early to 20-30s later and still rising, all of it on the
    training critical path.  ``mamba3-ship`` moves the finished tree once.

    python train.py --run-id 20260814T091500Z --run-root /workspace/observed-runs/...
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


# Defaults come from the environment so the image can set them once and a
# command line can still override any of them.
MODEL_ID = os.environ.get("MAMBA3_MODEL_ID", "state-spaces/mamba3-siso-443m")
MODEL_REVISION = os.environ.get(
    "MAMBA3_MODEL_REVISION", "efe60ce987877fcb72f089ab3a94478e99e2a6ae"
)
IMAGE_REF = os.environ.get("MAMBA3_IMAGE_REF", "private-runpod-wrapper")

TOKENIZER_ID = "meta-llama/Llama-3.1-8B"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "train"

SEED = 1234
MAX_TOKENS = 512
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
# Chosen from the observed gradient distribution of run hu1wjq02, not by feel.
# At 1.0 the clip fired on 100.00% of 14,946 steps with a mean applied scale of
# 0.278, which made it the update-magnitude controller rather than a safety
# valve and erased gradient magnitude as a signal.  8.0 sits just above p99.5
# (7.951), fires on 0.47% of steps, and applies a mean scale of 0.999.
GRAD_CLIP_NORM = 8.0
EPOCHS = 2

# Rows whose shapes are compiled during warmup.  The expensive compile is the
# first forward and backward; later rows add specializations cheaply.
WARMUP_ROWS = 25

OBSERVE_EVERY = 100  # epoch-local step; same rows observed on both passes
CHECKPOINT_EVERY = 0
DEFAULT_CAMERAS = "1,2,3,4,5,6"

WANDB_ENTITY = "ricks-holmberg-juiceb0xc0de"
WANDB_PROJECT = "mamba3-liveness"

# Where finished runs go.  /workspace does not outlive the pod, so this is the
# only durable copy and the upload is part of the run rather than a follow-up.
HF_DATASET = "juiceb0xc0de/mmb3-slt-6xcam-results"

# Modules whose capture window is pinned per observation point.  Each observer
# derives its expected filenames from these globals and CAPTURE_WINDOW is
# materialized at import, so all three have to move together.
WINDOW_MODULES = (
    "training_time_observer",
    "training_pressure_observer",
    "training_optimizer_delivery_observer",
    "training_memory_path_camera",
    "training_memory_path_observer",
    "training_embedding_learning_observer",
    "training_layer_transaction_observer",
    "training_output_loss_camera",
    "training_jacobian_actions_observer",
)


def locker_path() -> Path:
    """Where cameras come from.

    A locker on /workspace wins over the one baked into the image.  That
    preserves the property the Modal Volume gave: a camera can be edited on a
    live pod without rebuilding anything.
    """
    explicit = os.environ.get("MAMBA3_LOCKER")
    if explicit:
        return Path(explicit)
    workspace = Path("/workspace/camera-locker")
    if workspace.is_dir():
        return workspace
    return Path(os.environ.get("MAMBA3_LOCKER_BAKED", "/opt/mamba3/camera-locker"))


def _prepare_sys_path(locker: Path):
    support = locker / "support"
    # The locker root carries shared runner modules such as timing.py.
    for entry in (support / "analysis", support, locker):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_camera_module(locker: Path, camera_id):
    path = locker / f"camera-{camera_id}.py"
    if not path.is_file():
        raise RuntimeError(f"camera {camera_id} not in locker: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = f"trainer_camera_{camera_id}_{digest[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, digest


def _pin_window(step):
    """Point every observer's capture window at this single step.

    CAPTURE_WINDOW is materialized at import time from the two constants, and
    cameras read it into their own state at construction, so all three have to
    move together and this has to run before create_camera.
    """
    for name in WINDOW_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if hasattr(module, "CAPTURE_START_STEP"):
            module.CAPTURE_START_STEP = step
        if hasattr(module, "CAPTURE_END_STEP"):
            module.CAPTURE_END_STEP = step
        if hasattr(module, "CAPTURE_WINDOW"):
            module.CAPTURE_WINDOW = frozenset({step})


def _tokenize(tokenizer, bos_id, eos_id, row):
    text = f"Question:\n{row['question']}\n\nAnswer:\n{row['answer']}"
    body = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [bos_id] + body[: MAX_TOKENS - 2] + [eos_id]


def _fire(cameras, hook, ctx):
    for camera in cameras:
        method = getattr(camera, hook, None)
        if method is None:
            continue
        try:
            method(ctx)
        except Exception as error:
            raise RuntimeError(
                f"camera {getattr(camera, 'camera_id', '?')} failed in {hook} "
                f"at step {ctx.step}"
            ) from error


def _close_all(cameras, ctx):
    """Close every camera without letting cleanup replace the real error."""
    for camera in cameras:
        method = getattr(camera, "close_step", None)
        if method is None:
            continue
        try:
            method(ctx)
        except Exception as error:
            print(f"  close_step failed: {error!r}", flush=True)


def build_scene(model_id: str, model_revision: str):
    """Load the model and compile kernels, leaving the weights at step 0.

    Modal took a GPU memory snapshot after this and restored it per run.  There
    is no snapshot here, so this runs at every start.  What it does is
    unchanged: forward and backward only, no optimizer in existence, no step
    taken, so the parameters cannot have moved.  Gradients are the only thing
    produced and they are dropped immediately, and the RNG state is restored
    afterwards so the warmup leaves no trace beyond compiled kernels.
    """
    import random

    import huggingface_hub
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    started = time.perf_counter_ns()
    random.seed(SEED)
    torch.manual_seed(SEED)

    def milestone(label, since):
        print(
            f"[setup] {label:<26} {(time.perf_counter_ns() - since) / 1e9:7.2f}s",
            flush=True,
        )
        return time.perf_counter_ns()

    mark = started
    model_path = huggingface_hub.snapshot_download(
        repo_id=model_id, revision=model_revision
    )
    mark = milestone("model downloaded", mark)
    model = MambaLMHeadModel.from_pretrained(
        model_path, device="cuda", dtype=torch.bfloat16
    )
    model.train()
    mark = milestone("model on gpu", mark)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    assert bos_id is not None and eos_id is not None
    mark = milestone("tokenizer", mark)

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
    rows = [dataset[i] for i in range(len(dataset))]
    del dataset
    mark = milestone(f"dataset ({len(rows)} rows)", mark)

    rng = (
        random.getstate(),
        torch.get_rng_state().clone(),
        [s.clone() for s in torch.cuda.get_rng_state_all()],
    )
    for index in range(min(WARMUP_ROWS, len(rows))):
        ids = _tokenize(tokenizer, bos_id, eos_id, rows[index])
        tensor = torch.tensor(ids, dtype=torch.long, device="cuda")
        inputs = tensor[:-1].unsqueeze(0)
        targets = tensor[1:].unsqueeze(0)
        model.zero_grad(set_to_none=True)
        logits = model(inputs).logits.float()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        loss.backward()
        if index == 0:
            mark = milestone("first fwd+bwd (compile)", mark)
    model.zero_grad(set_to_none=True)

    # The warmup is not allowed to leave a trace beyond compiled kernels.
    random.setstate(rng[0])
    torch.set_rng_state(rng[1])
    torch.cuda.set_rng_state_all(rng[2])
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mark = milestone(f"warmed {WARMUP_ROWS} shapes", mark)

    seconds = (time.perf_counter_ns() - started) / 1e9
    print(f"[setup] scene ready at step 0, {seconds:.1f}s total", flush=True)
    return SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        bos_id=bos_id,
        eos_id=eos_id,
        rows=rows,
        build_seconds=seconds,
    )


def ship(run_root: Path, run_id: str, repo: str, include_checkpoints: bool) -> bool:
    """Upload the finished run to HuggingFace. Reads files, never moves them.

    Runs at the end of training rather than per observation point.  The Modal
    trainer pushed each point inline and every one was a separate commit into a
    growing dataset repo, so the cost climbed as the run went: ~6s early,
    20-30s later and still rising, all of it on the training critical path.
    One upload at the end costs the same bytes and none of the compounding.

    This matters more here than it did on Modal, because /workspace does not
    outlive the pod.  Returns True on success; on failure it says exactly what
    to re-run and leaves every local file untouched, so a retry is free.
    """
    files = [
        p for p in run_root.rglob("*")
        if p.is_file()
        and not (not include_checkpoints
                 and "checkpoints" in p.relative_to(run_root).parts)
    ]
    if not files:
        print(f"nothing to upload in {run_root}", flush=True)
        return False

    total = sum(p.stat().st_size for p in files)
    repo_path = f"observed-runs/{run_id}"
    print(f"\nuploading {len(files)} files, {total / 1e9:.2f} GB "
          f"-> {repo}/{repo_path}", flush=True)

    from huggingface_hub import HfApi

    api = HfApi()
    ignore = None if include_checkpoints else ["checkpoints/**"]

    # Three tries, exponential from 5s, capped at 2m, jittered so a retry does
    # not land on the same second as anything else.
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            api.upload_folder(
                repo_id=repo,
                repo_type="dataset",
                folder_path=str(run_root),
                path_in_repo=repo_path,
                ignore_patterns=ignore,
                commit_message=f"{run_id}: {len(files)} files",
            )
        except Exception as error:
            print(f"upload attempt {attempt}/3 failed after "
                  f"{time.perf_counter() - started:.0f}s: {error!r}", flush=True)
            if attempt == 3:
                import random
                print("\n" + "=" * 67, flush=True)
                print(" UPLOAD FAILED. Nothing local was modified.", flush=True)
                print(f" the run is intact at {run_root}", flush=True)
                print(" retry with:", flush=True)
                print(f"   mamba3 --ship-only {run_id}", flush=True)
                print(" DO NOT KILL THIS POD until it has shipped.", flush=True)
                print("=" * 67 + "\n", flush=True)
                return False
            import random
            delay = min(5.0 * (2 ** (attempt - 1)), 120.0) + random.uniform(0, 2)
            print(f"retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
        else:
            print(f"uploaded in {(time.perf_counter() - started) / 60:.1f}m", flush=True)
            print(f"https://huggingface.co/datasets/{repo}/tree/main/{repo_path}",
                  flush=True)
            return True
    return False


def train(args) -> dict:
    import torch
    import torch.nn.functional as F
    import wandb

    locker = locker_path()
    _prepare_sys_path(locker)

    camera_ids = [int(v) for v in args.cameras.split(",") if v.strip()]
    observe_set = {int(s) for s in args.observe_steps.split(",") if s.strip()}
    observe_every = args.observe_every or OBSERVE_EVERY

    scene_parts = build_scene(args.model_id, args.model_revision)
    model = scene_parts.model
    tokenizer = scene_parts.tokenizer
    bos_id, eos_id = scene_parts.bos_id, scene_parts.eos_id
    rows = scene_parts.rows
    n_rows = len(rows) if not args.max_steps else min(len(rows), args.max_steps)

    # Built after the scene, so lr and weight decay are never baked into it.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    started = time.perf_counter_ns()
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = Path(args.run_root) if args.run_root else (
        Path(os.environ.get("MAMBA3_OUTPUT_ROOT", "/workspace/observed-runs")) / run_id
    )
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = run_root / "checkpoints"
    if args.checkpoint_every:
        ckpt_root.mkdir(parents=True, exist_ok=True)

    camera_modules = {}
    for camera_id in camera_ids:
        module, digest = _load_camera_module(locker, camera_id)
        camera_modules[camera_id] = module
        print(f"camera {camera_id} loaded sha256={digest[:12]}", flush=True)

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"observed-{run_id}",
        config={
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "lr": args.lr,
            "scheduler": "none",
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip,
            "epochs": args.epochs,
            "rows": n_rows,
            "shuffle": False,
            "observe_every": observe_every,
            "observe_steps": sorted(observe_set),
            "checkpoint_every": args.checkpoint_every,
            "camera_ids": camera_ids,
            "image_ref": IMAGE_REF,
            "seed": SEED,
            "note": args.note,
            "warm_snapshot": False,
            "warmup_rows": WARMUP_ROWS,
            "platform": "runpod",
            "camera_locker": str(locker),
            "run_root": str(run_root),
        },
    )

    # grad_clip travels on the scene because Camera 3's observer runs its own
    # clip cross-check and needs the threshold the trainer actually used.  Left
    # to its own default it polices 1.0, and every step where the real clip does
    # not fire fails the check.
    scene = SimpleNamespace(
        model=model, optimizer=optimizer, grad_clip=args.grad_clip
    )
    global_step = 0
    observations = []
    window: list[float] = []
    losses: list[float] = []

    observe_set = {s for s in observe_set if s <= n_rows}
    if observe_set:
        schedule = (
            f"  observe {len(observe_set)} named local steps {sorted(observe_set)}\n"
            f"  ({len(observe_set)} points per epoch, cameras {camera_ids})\n"
        )
    else:
        schedule = (
            f"  observe every {observe_every} local steps "
            f"({n_rows // observe_every} points per epoch, cameras {camera_ids})\n"
        )
    ckpt_line = (
        f"  checkpoint every {args.checkpoint_every} local steps, plus epoch end\n"
        if args.checkpoint_every else "  checkpointing OFF\n"
    )
    print(
        f"\nrun {run_id}\n"
        f"  {args.model_id} @ {args.model_revision[:12]}\n"
        f"  {n_rows} rows x {args.epochs} epochs = {n_rows * args.epochs} steps\n"
        f"  lr {args.lr} constant, no scheduler, no shuffle\n"
        f"  weight decay {args.weight_decay}, grad clip {args.grad_clip}\n"
        f"{schedule}"
        f"{ckpt_line}"
        f"  cameras from {locker}\n"
        f"  all output to {run_root}\n"
        f"  heartbeat every {args.log_every} steps\n",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter_ns()
        print(f"=== epoch {epoch}/{args.epochs} ===", flush=True)
        for local_step in range(1, n_rows + 1):
            row_index = local_step - 1
            global_step += 1
            observed = (
                local_step in observe_set if observe_set
                else (local_step % observe_every) == 0
            )

            ids = _tokenize(tokenizer, bos_id, eos_id, rows[row_index])
            ids_tensor = torch.tensor(ids, dtype=torch.long, device="cuda")
            inputs = ids_tensor[:-1].unsqueeze(0)
            targets = ids_tensor[1:].unsqueeze(0)

            cameras = []
            out_dirs = {}
            if observed:
                _pin_window(global_step)
                point_root = run_root / f"step_{global_step:06d}"
                for camera_id in camera_ids:
                    out_dir = point_root / f"camera-{camera_id}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cameras.append(
                        camera_modules[camera_id].create_camera(scene, str(out_dir))
                    )
                    out_dirs[str(camera_id)] = str(out_dir)

            ctx = SimpleNamespace(
                step=global_step,
                ids=ids,
                tokens=len(ids),
                n_tokens=int(inputs.shape[-1]),
                dataset_row=row_index,
                input_ids=inputs,
                targets=targets,
                loss_value=None,
                grad_norm=None,
            )

            step_started = time.perf_counter_ns()
            try:
                _fire(cameras, "start_step", ctx)
                optimizer.zero_grad(set_to_none=True)
                _fire(cameras, "before_forward", ctx)
                logits = model(inputs).logits.float()
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
                ctx.loss_value = float(loss.detach())
                _fire(cameras, "after_forward", ctx)

                loss.backward()
                _fire(cameras, "after_backward", ctx)

                _fire(cameras, "before_clip", ctx)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                ctx.grad_norm = float(grad_norm)
                _fire(cameras, "after_clip", ctx)

                optimizer.step()
                _fire(cameras, "after_optimizer_step", ctx)
                _fire(cameras, "post_update", ctx)
                _fire(cameras, "finalize_step", ctx)
            except BaseException:
                _close_all(cameras, ctx)
                raise
            else:
                _fire(cameras, "close_step", ctx)

            seconds = (time.perf_counter_ns() - step_started) / 1e9
            wandb.log(
                {
                    "epoch": epoch,
                    "local_step": local_step,
                    "row_index": row_index,
                    "loss": ctx.loss_value,
                    "grad_norm": ctx.grad_norm,
                    "grad_norm_clipped": min(ctx.grad_norm, args.grad_clip),
                    "tokens": ctx.tokens,
                    "lr": args.lr,
                    "seconds": seconds,
                    "observed": int(observed),
                },
                step=global_step,
            )

            if observed:
                run_meta = {
                    "run_id": run_id,
                    "epoch": epoch,
                    "local_step": local_step,
                    "global_step": global_step,
                    "row_index": row_index,
                    "model_id": args.model_id,
                    "model_revision": args.model_revision,
                    "image_ref": IMAGE_REF,
                    "lr": args.lr,
                    "grad_clip_norm": args.grad_clip,
                    "camera_ids": camera_ids,
                    "capture_start_step": global_step,
                    "capture_end_step": global_step,
                    "steps": [
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "local_step": local_step,
                            "row_index": row_index,
                            "loss": ctx.loss_value,
                            "grad_norm": ctx.grad_norm,
                            "seconds": seconds,
                        }
                    ],
                }
                for camera in cameras:
                    finalize = getattr(camera, "finalize_run", None)
                    if finalize is not None:
                        finalize(run_meta)

                observations.append(
                    {
                        "epoch": epoch,
                        "local_step": local_step,
                        "global_step": global_step,
                        "row_index": row_index,
                        "loss": ctx.loss_value,
                        "grad_norm": ctx.grad_norm,
                        "seconds": seconds,
                        "out_dirs": out_dirs,
                    }
                )
                print(
                    f"observed e{epoch} local={local_step} "
                    f"global={global_step} loss={ctx.loss_value:.6f} "
                    f"seconds={seconds:.2f}",
                    flush=True,
                )

            # Keyed on the epoch-local step, not the global one, so both passes
            # checkpoint at the same offsets and therefore the same rows.  End
            # of epoch always gets one: it is the entry point for replaying the
            # following pass.
            end_of_epoch = local_step == n_rows
            if args.checkpoint_every and (
                local_step % args.checkpoint_every == 0 or end_of_epoch
            ):
                path = ckpt_root / f"step_{global_step:06d}.pt"
                torch.save(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "local_step": local_step,
                        "row_index": row_index,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "grad_clip_norm": args.grad_clip,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    path,
                )
                print(
                    f"  [ckpt] {path.name}  {path.stat().st_size / 1e9:.2f} GB",
                    flush=True,
                )

            window.append(seconds)
            losses.append(ctx.loss_value)
            if local_step % args.log_every == 0 or local_step == 1:
                rate = len(window) / max(sum(window), 1e-9)
                remaining = (n_rows - local_step) + (args.epochs - epoch) * n_rows
                eta_min = remaining / max(rate, 1e-9) / 60.0
                print(
                    f"e{epoch} {local_step:>5}/{n_rows} "
                    f"(g {global_step:>5}/{n_rows * args.epochs}) "
                    f"loss={ctx.loss_value:6.4f} "
                    f"avg={sum(losses) / len(losses):6.4f} "
                    f"grad={ctx.grad_norm:6.3f} "
                    f"{rate:5.2f} step/s  eta {eta_min:5.1f}m",
                    flush=True,
                )
                window.clear()
                losses.clear()

        print(
            f"=== epoch {epoch} done in "
            f"{(time.perf_counter_ns() - epoch_started) / 1e9 / 60:.1f}m ===\n",
            flush=True,
        )

    summary = {
        "run_id": run_id,
        "run_root": str(run_root),
        "platform": "runpod",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "image_ref": IMAGE_REF,
        "epochs": args.epochs,
        "rows": n_rows,
        "global_steps": global_step,
        "camera_ids": camera_ids,
        "camera_locker": str(locker),
        "observe_every": observe_every,
        "observe_steps": sorted(observe_set),
        "checkpoint_every": args.checkpoint_every,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip,
        "scheduler": "none",
        "shuffle": False,
        "seed": SEED,
        "warm_snapshot": False,
        "setup_seconds": scene_parts.build_seconds,
        "observations": observations,
        "total_seconds": (time.perf_counter_ns() - started) / 1e9,
    }
    (run_root / "run_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )
    wandb.finish()

    print(f"\nrun {run_id} complete")
    print(f"  {global_step} steps, {len(observations)} observation points")
    print(f"  {summary['total_seconds'] / 60:.1f}m")

    # Training succeeded, so the capture is complete and safe to publish.  A
    # run that died partway never reaches this line, deliberately: a
    # half-written capture with no completion marker landing in the dataset is
    # a provenance problem that outlives the pod it came from.
    if args.upload:
        ship(run_root, run_id, args.repo, args.include_checkpoints)
    else:
        print(f"  upload skipped. ship later with:  mamba3 --ship-only {run_id}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--cameras", default=DEFAULT_CAMERAS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--observe-every", type=int, default=0)
    parser.add_argument(
        "--observe-steps", default="",
        help="comma-separated epoch-local steps; empty uses --observe-every")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--note", default="")
    parser.add_argument("--repo", default=HF_DATASET)
    parser.add_argument("--no-upload", dest="upload", action="store_false",
                        help="finish and leave the run on disk")
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="checkpoints are large and replayable; excluded by default")
    parser.add_argument("--ship-only", default="",
                        help="upload an existing run id and exit, no training")
    args = parser.parse_args()

    # Retry path for an upload that failed after a run that succeeded.  It
    # never touches the trainer, so a completed run can never be repeated by
    # accident while trying to publish it.
    if args.ship_only:
        root = Path(os.environ.get("MAMBA3_OUTPUT_ROOT", "/workspace/observed-runs"))
        run_root = Path(args.run_root) if args.run_root else root / args.ship_only
        if not run_root.is_dir():
            print(f"no such run: {run_root}", file=sys.stderr)
            available = sorted(p.name for p in root.glob("*") if p.is_dir())
            print(f"available: {available}", file=sys.stderr)
            return 2
        return 0 if ship(run_root, args.ship_only, args.repo,
                         args.include_checkpoints) else 1

    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
