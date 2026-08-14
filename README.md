# Private RunPod wrapper, Mamba-3 443M observed run

Ports the Modal six-camera trainer to a RunPod pod. The science is unchanged;
only the plumbing Modal used to own has been replaced.

Nothing here has been built or pushed. No image exists, no pod exists, no paid
compute has run.

## File inventory

| file | what it is |
|---|---|
| `train.py` | the whole thing: trains, then uploads. Installed as `mamba3` |
| `Dockerfile` | base image plus tvm-ffi 0.1.10, wandb, sshd, boo, cameras |
| `build.sh` | stage sources, build via attempt + BuildKit, always delete `src/` |
| `entrypoint.sh` | PID 1: sshd, keys, keepalive. Never starts training |
| `.dockerignore` | deny-all allow-list |

One command, not a launcher and a shipper and a status tool. `mamba3` trains
and then uploads the finished run, the way a trainer does.

`src/` does not appear because it does not persist. `build.sh` stages the
camera and observer sources into it, builds, and removes it on any exit path
including failure and Ctrl-C. A second copy of an observer living on disk is
how a tree ends up with two versions and no way to tell which one ran.

## Checks completed

- `bash -n` passes on `build.sh`, `entrypoint.sh`, `bin/mamba3-run`
- `py_compile` passes on `train.py`, `bin/mamba3-ship`
- all **18** source files `build.sh` stages exist in `~/mamba3-project`, 0 missing
- boo v0.6.4 linux/amd64 release asset exists, SHA-256 pinned in the Dockerfile
  and verified with `sha256sum -c` during the build
- build-time gate asserts `tvm_ffi==0.1.10`, `tilelang==0.1.8`,
  `torch==2.11.0+cu128`, exactly 6 cameras present, and `compileall` over
  everything staged

Not checked, and cannot be from here: importing the observers, which several
reach CUDA to do. `mamba3-run` does that as a preflight on the pod, before any
GPU time is committed.

## The tvm-ffi question, answered

The base image carries `apache-tvm-ffi==0.1.8.post2` and that is correct for
training and prefill. Camera 4 replays token by token through `Mamba3.step()`,
which reaches NVIDIA CuteDSL, and that path needs a newer FFI API. The working
window is 0.1.10 to 0.1.11; 0.1.12 and later break TileLang coexistence.

The Modal image applied that upgrade at deploy time
(`simplified/trainer.py:152`), so **the runtime that produced every existing
six-camera capture was 0.1.10**, even though the recorded `image_ref` in every
manifest names the 0.1.8.post2 base. This wrapper bakes the same upgrade, so
the recorded image reference will now be true rather than approximately true.

## Build

Not runnable until a private registry target exists.

```
cd ~/443-ssm/runpod-private
PRIVATE_IMAGE=<registry>/<name>:<tag> ./build.sh
```

Local build without pushing, to prove the image assembles:

```
PRIVATE_IMAGE=mamba3-443m:local ./build.sh --no-push
```

`build.sh` wraps the exact `attempt` invocation from the brief. Do not add
`--no-cache`: BuildKit's layer cache is what makes a retry a resume.

## RunPod template settings

GPU, container disk, and volume size are chosen per pod at launch time and are
deliberately not fixed here.

| setting | value |
|---|---|
| Container image | `ghcr.io/juiceb0xc0de/mamba-view:<tag>` |
| Registry credentials | required, the package is private |
| Volume mount path | `/workspace` |
| Expose TCP port | 22 |
| Docker command | leave empty, the ENTRYPOINT handles it |
| Env `PUBLIC_KEY` | your SSH public key |
| Env `HF_TOKEN` | runtime only |
| Env `WANDB_API_KEY` | runtime only |
| Env `HF_HOME` | `/workspace/.hf` so the model cache survives a restart |

## Shipping

`mamba3-run` attempts the upload itself when the run ends, on success or
failure, because partial artifacts from a run that died at hour three are worth
more than nothing. The attempt is exactly that. It reads local files and never
moves or deletes one, so a failure costs a retry and nothing else, and it says
so in a block you cannot miss in scrollback:

```
 UPLOAD FAILED           exit status 1
 nothing local was modified. the run is intact at:
   /workspace/observed-runs/<run-id>
 retry whenever you like:
   mamba3-ship <run-id>
 DO NOT KILL THIS POD until it has shipped.
```

Then it drops you into a shell in the same session, still holding the run.

## Unresolved blockers

**1. The image has never been built.** `build.sh` has not run, so nothing has
proven that the base image plus tvm-ffi 0.1.10 plus boo actually assembles. A
`--no-push` build answers that without touching a registry or a pod.

**2. Warmup runs at every start.** There is no GPU snapshot here. Every launch
pays the Triton compile for 25 shapes, and the duration at 443M is unknown.
Not lost work, just not the seconds Modal made it look like.

**3. Cameras were written against 187M geometry.** If any of them holds a
hardcoded dimension it will surface at the first observation point rather than
at startup, since the preflight imports cameras but does not attach them to the
model. Whether that is worth a short run first is a judgment call about run
budget, not a technical gate.
