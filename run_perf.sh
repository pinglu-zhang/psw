#!/usr/bin/env bash
# run_perf.sh - build psw and benchmark runtime/memory on FASTA inputs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_ROOT/build"
PSW_BIN="$BUILD_DIR/psw"

# Prefer user-requested names; fall back to existing MT-human.fa if needed.
TARGET_FA="$PROJECT_ROOT/test/MT-human.fa"
QUERY_FA="$PROJECT_ROOT/test/MT-orang.fa"
if [[ ! -f "$TARGET_FA" && -f "$PROJECT_ROOT/test/MT-human.fa" ]]; then
    TARGET_FA="$PROJECT_ROOT/test/MT-human.fa"
fi

# MODES is space-separated; easy to extend later, e.g. MODES="gg_pp gg_ps new_mode"
MODES_STR="${MODES:-gg_pp gg_ps}"
REPEAT="${REPEAT:-5}"
BAND="${BAND:--1}"

usage() {
    cat <<'EOF'
Usage: bash run_perf.sh [options]

Options:
  -w, --band INT      Band width passed to psw (-w). Default: -1
  -r, --repeat INT    Repeat count per mode. Default: 5
  -m, --modes STR     Space-separated modes, e.g. "gg_pp gg_ps"
  -h, --help          Show this help

Environment overrides also supported: BAND, REPEAT, MODES.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--band)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
            BAND="$2"; shift 2 ;;
        -r|--repeat)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
            REPEAT="$2"; shift 2 ;;
        -m|--modes)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
            MODES_STR="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

if [[ ! -f "$TARGET_FA" ]]; then
    echo "ERROR: target FASTA not found: $TARGET_FA" >&2
    exit 1
fi
if [[ ! -f "$QUERY_FA" ]]; then
    echo "ERROR: query FASTA not found: $QUERY_FA" >&2
    exit 1
fi
if [[ ! "$REPEAT" =~ ^[0-9]+$ ]] || [[ "$REPEAT" -lt 1 ]]; then
    echo "ERROR: REPEAT must be an integer >= 1" >&2
    exit 1
fi
if [[ ! "$BAND" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: BAND must be an integer" >&2
    exit 1
fi
if [[ ! -x /usr/bin/time ]]; then
    echo "ERROR: /usr/bin/time not found (required for memory measurement)" >&2
    exit 1
fi

echo "================================================================"
echo "  Building psw"
echo "================================================================"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_STANDARD=11 -Wno-dev
cmake --build "$BUILD_DIR" --config Release --parallel "$(nproc 2>/dev/null || echo 4)"

echo
echo "================================================================"
echo "  Performance benchmark"
echo "================================================================"
echo "target    : $TARGET_FA"
echo "query     : $QUERY_FA"
echo "repeats   : $REPEAT"
echo "band      : $BAND"
echo "modes     : $MODES_STR"
echo "binary    : $PSW_BIN"
echo

printf "%-10s | %-14s | %-14s\n" "mode" "avg_time(s)" "avg_mem(MB)"
printf "%-10s-+-%-14s-+-%-14s\n" "----------" "--------------" "--------------"

for mode in $MODES_STR; do
    sum_time="0"
    sum_rss_kb=0

    for ((i=1; i<=REPEAT; i++)); do
        cmd=("$PSW_BIN" -t "$mode" -w "$BAND" "$TARGET_FA" "$QUERY_FA")
        # printf "[run %d/%d] %s\n" "$i" "$REPEAT" "${cmd[*]}"

        read -r elapsed rss_kb < <({ /usr/bin/time -f "%e %M" \
            "${cmd[@]}" >/dev/null; } 2>&1)

        sum_time="$(awk -v a="$sum_time" -v b="$elapsed" 'BEGIN{printf "%.6f", a+b}')"
        sum_rss_kb=$((sum_rss_kb + rss_kb))
    done

    avg_time="$(awk -v s="$sum_time" -v n="$REPEAT" 'BEGIN{printf "%.6f", s/n}')"
    avg_mem_mb="$(awk -v kb="$sum_rss_kb" -v n="$REPEAT" 'BEGIN{printf "%.3f", (kb/n)/1024.0}')"

    printf "%-10s | %-14s | %-14s\n" "$mode" "$avg_time" "$avg_mem_mb"
done

echo
echo "Done."

