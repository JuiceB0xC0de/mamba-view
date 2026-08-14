#!/usr/bin/env bash
# Build the private 443M RunPod image.
#
# Staging, build, and teardown in one script on purpose.  The image needs the
# camera and observer sources inside the build context, but a second copy of
# those files living on disk is how a tree ends up with two versions of the
# same observer and no way to tell which one ran.  So src/ exists only between
# the stage step and the build step, and the trap removes it on ANY exit:
# success, failure, or Ctrl-C.
#
# The build context is this directory rather than ~/mamba3-project because
# that tree carries roughly 1.1M lines of uncommitted NPZ captures, shot-run
# JSON and two 287 MB archives.  Every one of those would be tarred and shipped
# to the daemon on every build.  Here the contents are an explicit list.
#
# Retry policy comes from attempt-cli 1.1.1 wrapped around BuildKit.  Attempt
# reruns the same build command; BuildKit's layer cache is what makes that a
# resume rather than a restart, which is why --no-cache must never appear here.
# attempt is a local tool and is deliberately not installed into the image.
#
#     PRIVATE_IMAGE=registry/name:tag ./build.sh
#     PRIVATE_IMAGE=registry/name:tag ./build.sh --no-push   # local only
set -euo pipefail

SOURCE="${MAMBA3_SOURCE:-$HOME/mamba3-project}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/src"

# Three modes.  --stage exists because the real build runs on a GitHub runner,
# which checks out this repository and has no access to ~/mamba3-project, so
# src/ has to be committed rather than staged and thrown away.
#
#   --stage     write src/ and stop.  Run this, then commit, then push.
#   --no-push   stage, build locally, delete src/.  For proving it assembles.
#   (default)   stage, build, push, delete src/.
MODE=build
PUSH="--push"
case "${1:-}" in
  --stage)   MODE=stage ;;
  --no-push) PUSH="--load"; echo "local build only, image will NOT be pushed" ;;
  "")        ;;
  *)         echo "unknown flag: $1" >&2; exit 2 ;;
esac

if [[ "$MODE" == build && -z "${PRIVATE_IMAGE:-}" ]]; then
  echo "PRIVATE_IMAGE is unset. Set it to the registry target, e.g." >&2
  echo "  PRIVATE_IMAGE=ghcr.io/juiceb0xc0de/mamba-view:v1 ./build.sh" >&2
  echo "or stage only, for the GitHub Actions build:" >&2
  echo "  ./build.sh --stage" >&2
  exit 2
fi

# Only the local build path deletes src/.  In --stage mode it is the output.
if [[ "$MODE" == build ]]; then
  trap 'rm -rf "$DEST"' EXIT INT TERM
fi

# ---------------------------------------------------------------- stage
# Cameras 1-6 only.  7 and 8 are excluded deliberately: camera 7's output-loss
# replay and camera 8's Jacobian actions are not part of this run and would
# pull two more observer modules in for nothing.
CAMERAS=(camera-1.py camera-2.py camera-3.py camera-4.py camera-5.py camera-6.py)

# Locker root: cameras import these by bare name.
LOCKER_ROOT=(timing.py)

# One observer per camera, plus the second module camera 4 needs.
SUPPORT=(
  training_time_observer.py                  # camera 1
  training_pressure_observer.py              # camera 2
  training_optimizer_delivery_observer.py    # camera 3
  training_memory_path_camera.py             # camera 4
  training_memory_path_observer.py           # camera 4
  training_embedding_learning_observer.py    # camera 5
  training_layer_transaction_observer.py     # camera 6
)

ANALYSIS=(
  mamba3_core.py
  reference_recurrence.py
  memory_path_parity.py
  full_model_smoke.py
)

stage() {
  local from="$1" to="$2"
  [[ -f "$from" ]] || { echo "MISSING SOURCE: $from" >&2; exit 1; }
  cp "$from" "$to"
}

rm -rf "$DEST"
# The layout mirrors the Modal camera locker exactly, because train.py reuses
# the trainer's _prepare_sys_path() unchanged and it expects locker root,
# locker/support, and locker/support/analysis on sys.path in that order.
mkdir -p "$DEST/support/analysis"
for f in "${CAMERAS[@]}" "${LOCKER_ROOT[@]}"; do
  stage "$SOURCE/simplified/$f" "$DEST/$f"
done
for f in "${SUPPORT[@]}"; do
  stage "$SOURCE/$f" "$DEST/support/$f"
done
for f in "${ANALYSIS[@]}"; do
  stage "$SOURCE/analysis/$f" "$DEST/support/analysis/$f"
done

echo "staged from $SOURCE:"
find "$DEST" -type f | sed "s|$DEST|  src|" | sort
echo "  $(find "$DEST" -type f | wc -l | tr -d ' ') files, $(du -sh "$DEST" | cut -f1)"
echo

if [[ "$MODE" == stage ]]; then
  echo "staged only. src/ is the build input for GitHub Actions."
  echo "commit it and push; the workflow builds and pushes the image."
  echo
  echo "  git add -A && git commit -m 'restage cameras' && git push"
  exit 0
fi

# ---------------------------------------------------------------- build
# Exactly the command in the brief.  Do not add --no-cache: the layer cache is
# the resume mechanism that makes retrying worthwhile.
echo "building $PRIVATE_IMAGE"
attempt --attempts 3 \
  --retry-failing-status \
  --wait-min 5s \
  --wait-max 2m \
  --jitter 2s \
  exponential -- \
  docker buildx build \
    --platform linux/amd64 \
    --progress=plain \
    --tag "$PRIVATE_IMAGE" \
    $PUSH \
    "$HERE"

echo
echo "built $PRIVATE_IMAGE"
echo "src/ removed by trap"
