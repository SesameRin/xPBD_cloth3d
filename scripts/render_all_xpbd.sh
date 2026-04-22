#!/usr/bin/env bash
# Render every *_sim.npz under a results root to MP4 using the teammate's
# visualize_sim.py inside uipc_env. Skips NPZs whose expected output video
# already exists, and keeps going if a single render fails.
#
# Usage:
#   scripts/render_all_xpbd.sh [results_root] [extra visualize_sim.py flags...]
#
# Examples:
#   scripts/render_all_xpbd.sh
#   scripts/render_all_xpbd.sh xpbd_out/1_100_xpbd/results_xpbd
#   scripts/render_all_xpbd.sh xpbd_out/1_100_xpbd/results_xpbd --max_frames 60
#   FORCE=1 scripts/render_all_xpbd.sh      # re-render even if output exists

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VISUALIZE_SIM="/home/ula/CMU/pba-proj/cloth3d-ipc-xpbd/cloth3d_benchmark/visualize_sim.py"
CONDA_ENV="uipc_env"

# Defaults that match what you've been running by hand.
RENDERER="gpu"
CLOTH="both"
FORMAT="mp4"
SHOW_BODY="on"     # must stay in sync with visualize_sim.py's default to predict output name

RESULTS_ROOT="${1:-$REPO_ROOT/xpbd_out/1_100_xpbd/results_xpbd}"
shift || true
EXTRA_ARGS=("$@")

if [[ ! -f "$VISUALIZE_SIM" ]]; then
    echo "error: visualize_sim.py not found at $VISUALIZE_SIM" >&2
    exit 1
fi
if [[ ! -d "$RESULTS_ROOT" ]]; then
    echo "error: results root not found: $RESULTS_ROOT" >&2
    exit 1
fi

# Gather NPZs deterministically.
mapfile -t NPZS < <(find "$RESULTS_ROOT" -name "*_sim.npz" | sort)
N=${#NPZS[@]}
if (( N == 0 )); then
    echo "no *_sim.npz files found under $RESULTS_ROOT" >&2
    exit 1
fi

echo "results_root : $RESULTS_ROOT"
echo "renderer     : $RENDERER"
echo "cloth        : $CLOTH"
echo "format       : $FORMAT"
echo "extra        : ${EXTRA_ARGS[*]:-<none>}"
echo "total NPZs   : $N"
echo

# Predicts visualize_sim.py's default output filename so --skip-existing works
# without re-running the renderer. Must stay in sync with the naming rule in
# visualize_sim.py: "<stem>_<cloth>_<body_tag>_<renderer>.<ext>".
predict_output() {
    local npz="$1"
    local stem body_tag
    stem="$(basename "$npz" .npz)"
    body_tag="body"; [[ "$SHOW_BODY" == "off" ]] && body_tag="nobody"
    printf "%s/%s_%s_%s_%s.%s\n" \
        "$(dirname "$npz")" "$stem" "$CLOTH" "$body_tag" "$RENDERER" "$FORMAT"
}

failures=()
skipped=0
rendered=0
i=0
for npz in "${NPZS[@]}"; do
    i=$((i+1))
    out="$(predict_output "$npz")"
    tag="[$i/$N] $(basename "$(dirname "$npz")")"

    if [[ -z "${FORCE:-}" && -f "$out" ]]; then
        echo "$tag skip (exists): $(basename "$out")"
        skipped=$((skipped+1))
        continue
    fi

    echo "$tag render -> $(basename "$out")"
    if conda run -n "$CONDA_ENV" python "$VISUALIZE_SIM" \
            --npz "$npz" \
            --renderer "$RENDERER" \
            --cloth "$CLOTH" \
            --format "$FORMAT" \
            --show_body "$SHOW_BODY" \
            "${EXTRA_ARGS[@]}"; then
        rendered=$((rendered+1))
    else
        rc=$?
        echo "$tag FAILED (exit $rc)" >&2
        failures+=("$npz")
    fi
done

echo
echo "done. rendered=$rendered  skipped=$skipped  failed=${#failures[@]}  total=$N"
if (( ${#failures[@]} > 0 )); then
    echo "failed NPZs:"
    printf '  %s\n' "${failures[@]}"
    exit 1
fi
