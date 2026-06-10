#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Linux bench driver for the MorphiZen EP. Benchmarks an ONNX model from
# <workspace>/oga_models/<dir> using one of two host tools, captures the
# full stderr+stdout to a log under <script-dir>/_perf_logs/, and pipes
# the result through format_perf_report.py (sibling under this directory).
#
# Two tools, selected by flag:
#   (default)  onnxruntime_perf_test       -- per-Run() latency loop (kernel-level)
#   --oga      install/bin/model_benchmark -- TTFT + decode-TPS (production OGA path)
#
# Both default to the dynamic single-graph model.onnx (PR #212). The decoder-
# pipeline fixed-shape sub-models are still reachable via --shape static (perftest
# only; OGA fixed-pipeline isn't wired in this script -- invoke model_benchmark
# directly if you need it). Dynamic is the canonical Linux path going forward.
#
# Decode-only on dynamic shape for the PERFTEST path -- important caveat:
#   onnxruntime_perf_test silently rejects 3+ `-f` flags (parser quirk in the
#   prebuilt 1.25.1 binary; reproduced with 3/4 of {batch_size, sequence_length,
#   past_sequence_length, total_sequence_length}). We can only override 2 dims;
#   the other 2 fall back to ORT's default-1. The only shape that's still
#   internally consistent under that constraint is decode at past_len=1,
#   total_seq=1, seq=1. For prefill-shape perftest, --shape static is required.
#   OGA does NOT have this limitation -- it builds the input shapes correctly.
#
# Where to run this script:
#   This is host-tool-agnostic but expects an AMD GPU + the in-container
#   build artefacts ($WORKSPACE/install/{bin,lib}/...) to be reachable. The
#   two supported entry points are:
#     1. From inside the build container: `./docker/run.sh shell`, then run
#        this script directly.
#     2. From the host: forward via the sibling wrapper, which spins a
#        one-shot container with the GPU + bind mounts:
#          REL_BENCH=tools/perf-report/run_bench.sh \
#              tools/perf-report/docker_run_bench.sh --oga --prompt 128 ...
#
# Usage:
#   ./tools/perf-report/run_bench.sh                                  # perftest, dynamic seq=1, 60s
#   ./tools/perf-report/run_bench.sh --time 30                       # perftest, 30s window
#   ./tools/perf-report/run_bench.sh --oga                           # OGA, dynamic, prompt=128 gen=128 r=3 w=1
#   ./tools/perf-report/run_bench.sh --oga --prompt 128 --gen 128 --reps 3
#   ./tools/perf-report/run_bench.sh --oga --prompt 2048 --gen 128 --max-length 16384
#   ./tools/perf-report/run_bench.sh --shape static --seq 128 --seq 2048   # perftest, fixed-shape opt-in
#   ./tools/perf-report/run_bench.sh --model <other-dir>             # different model under oga_models/

set -euo pipefail

# ─── defaults ──────────────────────────────────────────────────────────────
MODEL_REL="Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml"
TOOL="perftest"             # perftest | oga
SHAPE="dynamic"             # dynamic | static (static = perftest fixed-pipeline opt-in)
SEQS=()                     # perftest: empty -> [1] for dynamic, [128 2048] for static
TIME_S=60                   # perftest only
MODE="none"                 # none | perf | debug -- toggles HIPDNN_EP_PERF / _DEBUG
TAG=""
# OGA defaults (match Windows model_benchmark.exe defaults from CLAUDE.md
# "Verified perf snapshot" section: -l 128 -g 128 -r 3 -w 1 captures the L=128
# row; user overrides -l to walk the L=2048 column etc.)
OGA_PROMPT=128
OGA_GEN=128
OGA_REPS=3
OGA_WARMUP=1
OGA_MAX_LENGTH=""           # empty -> let model_benchmark default to prompt+gen
                            # NOTE: CLAUDE.md "For dynamic-shape models, do NOT pass -ml"
                            # -- inflates peak WS without changing TPS/TTFT.

# ─── arg parse ────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --model)        MODEL_REL="$2"; shift 2 ;;
        --oga)          TOOL="oga"; shift ;;
        --shape)        SHAPE="$2"; shift 2 ;;
        --seq)          SEQS+=("$2"); shift 2 ;;
        --time|-t)      TIME_S="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --tag)          TAG="$2"; shift 2 ;;
        --prompt|-l)    OGA_PROMPT="$2"; shift 2 ;;
        --gen|-g)       OGA_GEN="$2"; shift 2 ;;
        --reps|-r)      OGA_REPS="$2"; shift 2 ;;
        --warmup|-w)    OGA_WARMUP="$2"; shift 2 ;;
        --max-length|-ml) OGA_MAX_LENGTH="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,43p' "$0"; exit 0 ;;
        *)  echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "$TOOL"  in perftest|oga)    ;; *) echo "bad --tool (internal): $TOOL"  >&2; exit 1 ;; esac
case "$SHAPE" in dynamic|static)  ;; *) echo "bad --shape: $SHAPE"            >&2; exit 1 ;; esac
case "$MODE"  in none|perf|debug) ;; *) echo "bad --mode: $MODE"              >&2; exit 1 ;; esac

# OGA currently only runs against the dynamic single-graph config. Static-
# pipeline OGA needs a different staging path (per-row sub-models, sliding
# window etc.) and isn't wired here; gate it explicitly so the user gets a
# clear error instead of silently running the perftest fallback.
if [ "$TOOL" = "oga" ] && [ "$SHAPE" = "static" ]; then
    echo "ERROR: --oga only supports --shape dynamic in this script." >&2
    echo "       For fixed-pipeline OGA runs, invoke model_benchmark directly with -i <model_dir>" >&2
    echo "       and the appropriate genai_config_p<L>m16384.json swapped in as genai_config.json." >&2
    exit 1
fi

# ─── locate repo root + workspace ─────────────────────────────────────────
# Walk upward from SCRIPT_DIR looking for `build.py` (the canonical project
# build script per CLAUDE.md). This decouples the script from its placement
# under the repo -- same logic works whether it lives at <repo>/oga/,
# <repo>/tools/perf-report/, or anywhere else. We deliberately don't shell
# out to `git rev-parse --show-toplevel` so the script remains usable in
# extracted source tarballs that have no .git/ directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR"
while [ "$SRC_DIR" != "/" ] && [ ! -f "$SRC_DIR/build.py" ]; do
    SRC_DIR="$(dirname "$SRC_DIR")"
done
if [ "$SRC_DIR" = "/" ]; then
    echo "ERROR: could not locate repo root from $SCRIPT_DIR" >&2
    echo "       (walked up looking for build.py; none found)" >&2
    exit 1
fi

# WORKSPACE defaults to the parent of the repo root, matching the convention
# `docker/run.sh` uses to choose the bind-mount root. Setting WORKSPACE
# explicitly on the command line overrides (e.g. when the build artefacts
# live somewhere other than `<repo>/../install/`).
: "${WORKSPACE:=$(cd "$SRC_DIR/.." && pwd)}"

# ─── env (mirrors docs/quick_start_linux.md "Open a container shell") ─────
ROOT="$WORKSPACE/install"
THEROCK_DIST="$WORKSPACE/therock-dist"
export LD_LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export THEROCK_DIST

case "$MODE" in
    perf)  export HIPDNN_EP_PERF=1 HIPDNN_EP_DEBUG=0 ;;
    debug) export HIPDNN_EP_PERF=1 HIPDNN_EP_DEBUG=1 ;;
    none)  export HIPDNN_EP_PERF=0 HIPDNN_EP_DEBUG=0 ;;
esac

EP_DLL="$ROOT/lib/libonnxruntime_morphizen_ep.so"
PERFTEST="$WORKSPACE/build/onnxruntime/Release/onnxruntime_perf_test"
MODEL_BENCHMARK="$ROOT/bin/model_benchmark"
[ -f "$EP_DLL" ] || { echo "missing: $EP_DLL"; exit 1; }
if [ "$TOOL" = "perftest" ]; then
    [ -x "$PERFTEST" ] || { echo "missing: $PERFTEST"; exit 1; }
else
    [ -x "$MODEL_BENCHMARK" ] \
        || { echo "missing: $MODEL_BENCHMARK -- rebuild with BUILD_OGA=1 ./docker/run.sh build"; exit 1; }
fi

# ─── PR #212 check: dynamic-shape support is reachable from HEAD ──────────
# `safe.directory='*'` disarms git's "dubious ownership" check that fires
# when the bench is run under a docker entrypoint that doesn't re-exec as
# the host UID. We treat any git failure (missing binary, unowned repo,
# detached worktree, ...) as "unknown" -- not "absent" -- so the script
# never falsely forces --shape static on a build that does have #212.
DYNSEQ_COMMIT="6b5cf01"   # feat: dynamic sequence length -- compile once, run any shape (#212)
GIT="git -c safe.directory=*"
pr212_state="unknown"
if $GIT -C "$SRC_DIR" rev-parse HEAD >/dev/null 2>&1; then
    if $GIT -C "$SRC_DIR" merge-base --is-ancestor "$DYNSEQ_COMMIT" HEAD 2>/dev/null; then
        pr212_state="present"
    else
        pr212_state="absent"
    fi
fi
case "$pr212_state" in
    present) echo "[pr212] dynamic-shape support present (commit $DYNSEQ_COMMIT reachable from HEAD)" ;;
    absent)
        if [ "$SHAPE" = "dynamic" ]; then
            echo "WARN: PR #212 ($DYNSEQ_COMMIT) NOT reachable from HEAD; --shape dynamic will likely fail"
            echo "      forcing --shape static (perftest path only)"
            if [ "$TOOL" = "oga" ]; then
                echo "ERROR: cannot fall back to static for --oga (see error above)." >&2
                exit 1
            fi
            SHAPE="static"
        fi ;;
    unknown) echo "[pr212] git unavailable; skipping reachability check (trusting --shape=$SHAPE)" ;;
esac

# ─── resolve model dir ───────────────────────────────────────────────────
# Models are NOT downloaded by this script. They must already exist at
# $WORKSPACE/oga_models/<MODEL_REL>/ with model.onnx + model.onnx.data +
# tokenizer.* + genai_config_MorphiZenEP.json. Typical staging step (one-
# time, host-side): `huggingface-cli download <repo> --local-dir
# $WORKSPACE/oga_models/<dir>`.
MODEL_DIR="$WORKSPACE/oga_models/$MODEL_REL"
[ -d "$MODEL_DIR" ] || { echo "missing model dir: $MODEL_DIR"; exit 1; }
MODEL_TAG="$(echo "$MODEL_REL" | tr '/' '_')"

# ─── helpers ─────────────────────────────────────────────────────────────
# Print the most recent per-op [PERF] table from a bench log. The table is
# emitted to stderr by op_profile_resolve_and_print (lib/Runtime/op_profile.cpp)
# at the end of every Compute() call, so a long generation produces hundreds
# of tables -- the existing tail -n N step strips them. We show only the LAST
# one (steady-state decode shape) inline; full history stays in the .log file.
#
# Self-gating on log content: no per-op tables in the log (e.g. --mode none,
# or a stale model.dll predating the flush_op_profile export) -> no output.
# That keeps the helper safe to call unconditionally.
print_last_op_profile_table() {
    local log="$1"
    [ -f "$log" ] || return 0
    # Locate the last two `[PERF] ===` borders -- they delimit the most recent
    # table. Done in two passes (grep for line numbers, sed for the slice)
    # to avoid the tac|awk|tac pipeline that, under `set -o pipefail`, can
    # propagate a SIGPIPE (exit 141) from an early awk exit and kill the
    # whole bench script right after printing the table.
    local borders top bot
    # `|| true` neutralizes grep's exit-1-on-no-match: under `set -o pipefail`
    # an empty match would otherwise make the substitution fail, and with
    # `set -e` the function would die before the `[ -n "$borders" ]` test.
    borders=$(grep -n '^\[PERF\] ===' "$log" | tail -2 | cut -d: -f1 || true)
    [ -n "$borders" ] || return 0
    top=$(printf '%s\n' "$borders" | head -1)
    bot=$(printf '%s\n' "$borders" | tail -1)
    # Need two DISTINCT borders for a complete block; a single border means
    # the run was killed mid-table or the model.dll predates the flush hook
    # and only the EP-side [PERF SUMMARY] block landed in the log.
    [ "$top" != "$bot" ] || return 0
    echo "  --- last per-op table ---"
    sed -n "${top},${bot}p" "$log" | sed 's/^/  /'
}

# Resolve the path to the locked-down Python report formatter. Lives next
# to this script under tools/perf-report/ (both files are tracked + ship
# together). Kept as a function so callers can `if [ -f "$(format_perf_report)" ]`
# to feature-detect, and so future relocation is a single edit.
format_perf_report() {
    echo "$SCRIPT_DIR/format_perf_report.py"
}

# Build (idempotently) a minimal model directory for OGA's dynamic single-
# graph path:
#   <dst>/model.onnx              -> symlink to <src>/model.onnx
#   <dst>/model.onnx.data         -> symlink to <src>/model.onnx.data
#   <dst>/tokenizer.json          -> symlink to <src>/tokenizer.json
#   <dst>/tokenizer_config.json   -> symlink to <src>/tokenizer_config.json
#   <dst>/genai_config.json       =  copy of <src>/genai_config_MorphiZenEP.json
#
# OGA picks up `genai_config.json` by name -- it has no flag to override the
# config filename -- so we copy the MorphiZenEP variant into place under the
# canonical name. Stays in <script-dir>/oga_dynamic_dirs/<tag> so it's
# adjacent to bench logs, writable by the host user, and easy to wipe
# (`rm -rf $SCRIPT_DIR/oga_dynamic_dirs/`).
stage_oga_dynamic_dir() {
    local src="$1" tag="$2"
    local dst="$SCRIPT_DIR/oga_dynamic_dirs/$tag"
    local cfg_src="$src/genai_config_MorphiZenEP.json"
    [ -f "$cfg_src" ] || {
        echo "ERROR: $cfg_src not found -- this model dir doesn't ship a dynamic OGA config." >&2
        return 1
    }
    [ -f "$src/model.onnx" ] || {
        echo "ERROR: $src/model.onnx not found -- this model dir doesn't ship a dynamic ONNX." >&2
        return 1
    }
    mkdir -p "$dst"
    local f
    # Use absolute symlinks so the dir is self-describing under `ls -la`.
    for f in model.onnx model.onnx.data tokenizer.json tokenizer_config.json; do
        if [ -e "$src/$f" ]; then
            ln -sfn "$src/$f" "$dst/$f"
        fi
    done
    # Always copy (not symlink) the config so we can safely diverge from it
    # (e.g. patch chunk_size) without mutating the user's pristine model dir.
    cp -f "$cfg_src" "$dst/genai_config.json"
    echo "$dst"
}

# ─── perftest path ────────────────────────────────────────────────────────
run_perftest() {
    local has_per_row has_single
    has_per_row=$(ls "$MODEL_DIR"/genai_config_p*m16384.json 2>/dev/null | wc -l)
    has_single=$([ -f "$MODEL_DIR/genai_config.json" ] && echo 1 || echo 0)
    case "$SHAPE" in
        static)
            [ "$has_per_row" -gt 0 ] \
                || { echo "--shape static but no genai_config_p*m16384.json"; exit 1; } ;;
        dynamic)
            [ "$has_single" = 1 ] \
                || { echo "--shape dynamic but no genai_config.json"; exit 1; } ;;
    esac

    if [ "${#SEQS[@]}" -eq 0 ]; then
        if [ "$SHAPE" = "static" ]; then SEQS=(128 2048); else SEQS=(1); fi
    fi

    # dynamic-shape sanity: warn loudly if user asked for seq>1 (perftest
    # parser only accepts 2 -f flags; seq>1 dynamic page-faults inside GQA).
    if [ "$SHAPE" = "dynamic" ]; then
        local s
        for s in "${SEQS[@]}"; do
            if [ "$s" -gt 1 ]; then
                echo "WARN: --shape dynamic with --seq $s will GPU page-fault (perftest 2-flag parser limit)"
                echo "      use --shape static for prefill-shape perftest, or --oga for prefill TPS"
            fi
        done
    fi

    local log_dir ts tag_suffix
    log_dir="$SCRIPT_DIR/_perf_logs"; mkdir -p "$log_dir"
    ts=$(date +%Y%m%d_%H%M%S)
    tag_suffix="${TAG:+_$TAG}"

    echo
    echo "=== $MODEL_TAG | perftest | shape=$SHAPE | seqs=${SEQS[*]} | mode=$MODE | t=${TIME_S}s ==="
    echo "    model dir: $MODEL_DIR"
    echo "    log dir  : $log_dir"
    echo

    local logs=()
    local seq model_file f_args=() log rc
    for seq in "${SEQS[@]}"; do
        case "$SHAPE" in
            static)
                model_file="$MODEL_DIR/prefill_p${seq}m16384.onnx"
                [ -f "$model_file" ] \
                    || { echo "  SKIP seq=$seq: $model_file not found"; continue; }
                f_args=() ;;
            dynamic)
                model_file="$MODEL_DIR/model.onnx"
                [ -f "$model_file" ] \
                    || { echo "  SKIP seq=$seq: $model_file not found"; continue; }
                f_args=(-f "batch_size:1" -f "sequence_length:$seq") ;;
        esac

        log="$log_dir/${MODEL_TAG}_perftest_${SHAPE}_seq${seq}_${MODE}${tag_suffix}_${ts}.log"
        echo "──────────────────────────────────────────────────────"
        echo " seq=$seq  $(basename "$model_file")  →  $(basename "$log")"
        echo "──────────────────────────────────────────────────────"

        SECONDS=0
        "$PERFTEST" \
            "${f_args[@]}" \
            --plugin_ep_libs "MorphiZenEP|$EP_DLL" \
            --plugin_eps     "MorphiZenEP" \
            -C "session.disable_cpu_ep_fallback|1" \
            -t "$TIME_S" -c 1 -s -I \
            "$model_file" >"$log" 2>&1
        rc=$?
        printf "  wall: %ds   exit=%d\n" "$SECONDS" "$rc"
        echo "  --- tail ---"
        tail -n 18 "$log" | sed 's/^/  /'
        echo "  ------------"
        logs+=("$log")
    done

    if [ "$MODE" = "perf" ] || [ "$MODE" = "debug" ]; then
        echo
        echo "=== headline P50 latency per row ==="
        local L
        for L in "${logs[@]}"; do
            [ -f "$L" ] || continue
            printf '%-60s ' "$(basename "$L")"
            grep -E "^P50 Latency:" "$L" | head -1 || echo "(no P50 line)"
        done
        echo
        echo "=== [PERF SUMMARY] (per-Compute brackets) ==="
        for L in "${logs[@]}"; do
            [ -f "$L" ] || continue
            if grep -q "^\[PERF SUMMARY\]" "$L"; then
                echo "--- $(basename "$L") ---"
                grep -E "^\[PERF SUMMARY\]" "$L" | tail -7
            fi
        done
        echo
        echo "=== last per-op [PERF] table per row ==="
        for L in "${logs[@]}"; do
            [ -f "$L" ] || continue
            if grep -q "^\[PERF\] ===" "$L"; then
                echo "--- $(basename "$L") ---"
                print_last_op_profile_table "$L"
            fi
        done
    fi
}

# ─── OGA path (model_benchmark on dynamic single-graph model.onnx) ────────
run_oga() {
    local oga_dir
    oga_dir="$(stage_oga_dynamic_dir "$MODEL_DIR" "$MODEL_TAG")" || exit 1

    local log_dir ts tag_suffix log
    log_dir="$SCRIPT_DIR/_perf_logs"; mkdir -p "$log_dir"
    ts=$(date +%Y%m%d_%H%M%S)
    tag_suffix="${TAG:+_$TAG}"
    log="$log_dir/${MODEL_TAG}_oga_dynamic_p${OGA_PROMPT}g${OGA_GEN}_${MODE}${tag_suffix}_${ts}.log"

    echo
    echo "=== $MODEL_TAG | OGA model_benchmark | dynamic | l=$OGA_PROMPT g=$OGA_GEN r=$OGA_REPS w=$OGA_WARMUP ==="
    echo "    model dir (staged): $oga_dir"
    echo "    log               : $log"
    echo "──────────────────────────────────────────────────────"

    local -a mb_args=(
        -i "$oga_dir"
        -l "$OGA_PROMPT" -g "$OGA_GEN"
        -r "$OGA_REPS"   -w "$OGA_WARMUP"
        -b 1
        -v
    )
    if [ -n "$OGA_MAX_LENGTH" ]; then
        mb_args+=(-ml "$OGA_MAX_LENGTH")
    fi

    # OGA's EP discovery looks for libonnxruntime_morphizen_ep.so in CWD on
    # Linux (the Windows equivalent of "next to onnxruntime-genai.dll"). cd
    # into install/lib so the load succeeds without --ep_library, which
    # CLAUDE.md flags as a double-registration crash on Windows.
    SECONDS=0
    ( cd "$ROOT/lib" && "$MODEL_BENCHMARK" "${mb_args[@]}" ) >"$log" 2>&1
    local rc=$?
    printf "  wall: %ds   exit=%d\n" "$SECONDS" "$rc"
    echo

    if [ "$rc" -ne 0 ]; then
        # Run failed -- format_perf_report.py would just print "no Batch size:"
        # warning, so dump the raw tail instead so the error / stack trace is
        # visible in the terminal (full log still on disk for forensics).
        echo "  --- raw tail (run failed) ---"
        tail -n 25 "$log" | sed 's/^/  /'
        echo "  ------------"
        return 0
    fi

    # Happy path: hand off to the locked-down formatter. format_perf_report.py
    # owns the entire report layout (banner + § 1 HEADLINE + § 2 BREAKDOWN +
    # § 3 PER-OP + § 4 DISTRIBUTION + [OGA] tail line). It self-gates on log
    # content -- missing [PERF SUMMARY] / per-op tables under --mode none, or
    # under an older model.dll predating the flush_op_profile export, will
    # silently render only the sections that have data.
    #
    # --indent 2 keeps the report flush with the rest of the bench output;
    # exit 2 (no model_benchmark stats) is treated as success here because
    # the `rc` check above already covered the actual failure path.
    python3 "$(format_perf_report)" "$log" --indent 2 || true
}

# ─── dispatch ─────────────────────────────────────────────────────────────
case "$TOOL" in
    perftest) run_perftest ;;
    oga)      run_oga      ;;
esac

echo
echo "done. logs in $SCRIPT_DIR/_perf_logs"
