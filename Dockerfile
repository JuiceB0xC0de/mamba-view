# Private RunPod wrapper for the 443M Mamba-3 observed run.
#
# This adds nothing to the science.  The base image is the proven atlas runtime
# and every pin inside it stays exactly where it is; this layer only supplies
# what Modal used to provide: a place to run, a way to reach it, and the two
# packages the Modal image built on top of the base at deploy time.
#
# Two additions are load-bearing and neither is cosmetic:
#
#   * apache-tvm-ffi==0.1.10.  The base carries 0.1.8.post2, which is correct
#     for training and prefill.  Camera 4 replays token by token through
#     Mamba3.step(), which reaches NVIDIA CuteDSL, and that path needs an FFI
#     API newer than 0.1.8.post2.  The working window is 0.1.10 to 0.1.11;
#     0.1.12 and later break TileLang coexistence.  The Modal image applied
#     this same upgrade at deploy time, so the runtime that produced every
#     existing six-camera capture was 0.1.10 even though the recorded image
#     reference names the 0.1.8.post2 base.  --no-deps because a free resolve
#     walks straight back to 0.1.12.
#
#   * wandb.  Not in the base; the Modal image added it for the trainer only.
#
# Cameras are baked here, which reverses the Modal rule that they never enter
# the image.  That rule existed because an image change invalidates a Modal GPU
# snapshot, and there is no snapshot on RunPod, so the cost it was avoiding
# does not exist.  The benefit it bought is preserved anyway: train.py prefers
# a locker at /workspace/camera-locker when one is present, so a camera can
# still be edited on a live pod without rebuilding anything.

FROM docker.io/juiceboxdocks/mamba3-atlas:1cde57a2d4981702add8c542ec75c7b2282b2ce3

# Camera 4's CuteDSL decode path.  See the note above before changing this.
RUN pip install --no-deps apache-tvm-ffi==0.1.10

# Trainer only; the snapshot path never needed it.
RUN pip install wandb

# openssh-server is the whole reason this wrapper exists: Modal gave a managed
# container lifecycle and RunPod gives a box, so the box has to stay up and be
# reachable on its own.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server rsync less procps \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh

# boo, not tmux.  Sessions survive disconnection either way, but boo has a
# scripting surface tmux does not: `ls --json` reports idle_ms per session,
# `peek --scrollback` reads a run's screen without attaching to it, and a
# missing session is exit code 3 rather than something to grep for.  mamba3-run
# uses all three, so the duplicate-launch check and the status output are exact
# instead of inferred.
#
# Version pinned and checksum verified.  This is a binary fetched over the
# network into an image that will hold credentials at run time; an unverified
# download here would be the weakest link in the whole build.
ARG BOO_VERSION=0.6.4
ARG BOO_SHA256=a49d4e1d1fd5fce4e7c71d7fedd862ad4f070767280f73cfa736b6ef84b96bd7
RUN curl -sSL -o /tmp/boo.tar.gz \
        "https://github.com/coder/boo/releases/download/v${BOO_VERSION}/boo-x86_64-linux.tar.gz" \
    && echo "${BOO_SHA256}  /tmp/boo.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/boo.tar.gz -C /usr/local/bin \
    && rm /tmp/boo.tar.gz \
    && chmod +x /usr/local/bin/boo \
    && boo version

# Camera locker, laid out exactly as the Modal Volume was, because train.py
# reuses _prepare_sys_path() unchanged and it expects these three directories
# on sys.path in this order: support/analysis, support, locker root.
COPY src/ /opt/mamba3/camera-locker/

# One command.  train.py trains and then uploads, the way a trainer does; it
# is not split across a launcher and a shipper.  Installed on PATH as `mamba3`
# so it reads as a command rather than a script path.
COPY train.py /opt/mamba3/train.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /opt/mamba3/train.py /usr/local/bin/entrypoint.sh \
    && ln -s /opt/mamba3/train.py /usr/local/bin/mamba3

# Baked locker is the fallback; /workspace/camera-locker wins when it exists.
# No secret may appear in any ENV or ARG here: HF_TOKEN, WANDB_API_KEY,
# PUBLIC_KEY and registry credentials are supplied at run time only.
# BOO_DIR is set explicitly rather than left to default.  boo falls back to
# /tmp/boo-<uid> when XDG_RUNTIME_DIR is unset, and an SSH login shell and the
# entrypoint can disagree about that, which would leave two daemons unable to
# see each other's sessions.  It stays on /tmp deliberately: a unix socket on
# the network-backed /workspace volume is asking for trouble.
ENV MAMBA3_LOCKER_BAKED=/opt/mamba3/camera-locker \
    MAMBA3_OUTPUT_ROOT=/workspace/observed-runs \
    MAMBA3_MODEL_ID=state-spaces/mamba3-siso-443m \
    MAMBA3_MODEL_REVISION=efe60ce987877fcb72f089ab3a94478e99e2a6ae \
    BOO_DIR=/tmp/boo \
    PYTHONUNBUFFERED=1

# Structural gate.  A broken image must fail here, not on a rented GPU.
# Imports of the observers themselves are deliberately NOT attempted at build
# time: several reach CUDA on import and the builder has no device, so they are
# checked by mamba3-run's preflight on the pod instead.
RUN python - <<'PYEOF'
import compileall, pathlib, sys
import tvm_ffi, tilelang, torch, wandb

print("torch    ", torch.__version__)
print("tilelang ", tilelang.__version__)
print("tvm_ffi  ", tvm_ffi.__version__)

assert tvm_ffi.__version__ == "0.1.10", "WRONG tvm-ffi: " + tvm_ffi.__version__
assert tilelang.__version__ == "0.1.8", "WRONG tilelang: " + tilelang.__version__
assert torch.__version__.startswith("2.11.0+cu128"), "WRONG torch: " + torch.__version__

locker = pathlib.Path("/opt/mamba3/camera-locker")
cameras = sorted(locker.glob("camera-*.py"))
assert len(cameras) == 6, f"expected 6 cameras, found {[c.name for c in cameras]}"

# Syntax-level check over everything that will be imported on the pod.
ok = compileall.compile_dir(str(locker), quiet=2, force=True)
ok &= compileall.compile_file("/opt/mamba3/train.py", quiet=2, force=True)
assert ok, "a staged source file failed to compile"

print("cameras  ", [c.name for c in cameras])
print("IMAGE GOOD")
PYEOF

WORKDIR /workspace

# Training must never be PID 1.  The entrypoint owns PID 1, starts sshd, and
# blocks forever, so a finished or crashed run leaves the pod alive and the
# shell usable rather than terminating the container.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
