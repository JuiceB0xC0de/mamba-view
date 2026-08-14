#!/usr/bin/env bash
# PID 1 for the pod.  Starts sshd, then blocks forever.
#
# It deliberately does not start training.  On Modal the container lifetime was
# the job's lifetime and that was correct there.  Here it would be a trap: a
# crash three hours in would take the pod down with it, along with the logs,
# the partial artifacts, and any chance of shipping them.  So PID 1 does
# nothing that can fail, training is launched by hand with mamba3-run, and the
# pod outlives the run in either direction.
#
# Nothing here reads a secret into a file that persists in an image layer.
# PUBLIC_KEY arrives as a runtime environment variable and is written to
# /root/.ssh/authorized_keys, which lives only in the container's writable
# layer.
set -euo pipefail

log() { echo "[entrypoint] $*"; }

mkdir -p /root/.ssh /workspace/observed-runs /workspace/logs
chmod 700 /root/.ssh

# boo keeps its socket here.  Created before anything can start a session, and
# on /tmp rather than the network-backed /workspace volume, because a unix
# socket on network storage is a bad time.
mkdir -p "${BOO_DIR:-/tmp/boo}"

# RunPod's convention: the pod's public key arrives in PUBLIC_KEY.
if [[ -n "${PUBLIC_KEY:-}" ]]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    log "installed PUBLIC_KEY"
else
    log "WARNING: PUBLIC_KEY unset, SSH will refuse key auth"
fi

# Host keys are generated per pod, never baked, so no image layer carries a
# private key.
ssh-keygen -A >/dev/null 2>&1
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

# An interactive SSH session gets its own environment, so the venv on PATH and
# the runtime secrets have to be planted where a login shell will read them.
{
    echo "export PATH=/opt/mamba3-venv/bin:\$PATH"
    echo "export PYTHONUNBUFFERED=1"
    for name in HF_TOKEN WANDB_API_KEY HF_HOME BOO_DIR MAMBA3_OUTPUT_ROOT \
                MAMBA3_MODEL_ID MAMBA3_MODEL_REVISION MAMBA3_LOCKER_BAKED; do
        value="${!name:-}"
        [[ -n "$value" ]] && printf 'export %s=%q\n' "$name" "$value"
    done
} > /etc/profile.d/mamba3.sh
chmod 600 /etc/profile.d/mamba3.sh

/usr/sbin/sshd
log "sshd up on port 22"

log "pod ready. training is NOT running."
log "  mamba3                       train, then upload to HuggingFace"
log "  mamba3 --no-upload           train and leave it on disk"
log "  mamba3 --ship-only <run-id>  upload an existing run, no training"
log ""
log "to survive an SSH drop, start it in a boo session:"
log "  boo new run -d -- mamba3     detached"
log "  boo attach run               reattach later"

# Block forever without spinning.  `sleep infinity` as PID 1 ignores signals in
# some shells, so wait on a background sleep instead and let signals through.
sleep infinity &
wait $!
