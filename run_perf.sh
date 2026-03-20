#!/usr/bin/env bash
# run_perf.sh - build psw and benchmark runtime/memory on FASTA inputs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_ROOT/build"
PSW_BIN="$BUILD_DIR/psw"

# DNA benchmark dataset (default)
TARGET_FA="${TARGET_FA:-$PROJECT_ROOT/test/MT-human.fa}"
QUERY_FA="${QUERY_FA:-$PROJECT_ROOT/test/MT-orang.fa}"

# Protein benchmark dataset (new)
PROT_TARGET_FA="${PROT_TARGET_FA:-$PROJECT_ROOT/test/protein-perf-target.fa}"
PROT_QUERY_FA="${PROT_QUERY_FA:-$PROJECT_ROOT/test/protein-perf-query.fa}"

# MODES is space-separated; easy to extend later, e.g. MODES="gg_pp gg_ps new_mode"
MODES_STR="${MODES:-gg_pp gg_ps gg2_pp gg2_ps gg3_pp gg3_ps gg3_sse_pp gg3_sse_ps}"
PROT_MODES_STR="${PROT_MODES:-$MODES_STR}"
REPEAT="${REPEAT:-1}"
BAND="${BAND:--1}"
RUN_DNA="${RUN_DNA:-1}"
RUN_PROTEIN="${RUN_PROTEIN:-1}"

usage() {
    cat <<'EOF'
Usage: bash run_perf.sh [options]

Options:
  -w, --band INT         Band width passed to psw (-w). Default: -1
  -r, --repeat INT       Repeat count per mode. Default: 1
  -m, --modes STR        Space-separated DNA modes, e.g. "gg_pp gg_ps"
  --protein-modes STR    Space-separated protein modes (default follows --modes)
  --dna-only             Run only DNA benchmark
  --protein-only         Run only protein benchmark
  -h, --help             Show this help

Environment overrides also supported:
  BAND, REPEAT, MODES, PROT_MODES,
  TARGET_FA, QUERY_FA, PROT_TARGET_FA, PROT_QUERY_FA,
  RUN_DNA, RUN_PROTEIN.
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
        --protein-modes)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }
            PROT_MODES_STR="$2"; shift 2 ;;
        --dna-only)
            RUN_DNA=1
            RUN_PROTEIN=0
            shift ;;
        --protein-only)
            RUN_DNA=0
            RUN_PROTEIN=1
            shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

if [[ ! "$REPEAT" =~ ^[0-9]+$ ]] || [[ "$REPEAT" -lt 1 ]]; then
    echo "ERROR: REPEAT must be an integer >= 1" >&2
    exit 1
fi
if [[ ! "$BAND" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: BAND must be an integer" >&2
    exit 1
fi
if [[ ! "$RUN_DNA" =~ ^[01]$ ]] || [[ ! "$RUN_PROTEIN" =~ ^[01]$ ]]; then
    echo "ERROR: RUN_DNA and RUN_PROTEIN must be 0 or 1" >&2
    exit 1
fi
if [[ "$RUN_DNA" -eq 0 && "$RUN_PROTEIN" -eq 0 ]]; then
    echo "ERROR: both DNA and protein benchmarks are disabled" >&2
    exit 1
fi
if [[ ! -x /usr/bin/time ]]; then
    echo "ERROR: /usr/bin/time not found (required for memory measurement)" >&2
    exit 1
fi

if [[ "$RUN_DNA" -eq 1 ]]; then
    if [[ ! -f "$TARGET_FA" ]]; then
        echo "ERROR: DNA target FASTA not found: $TARGET_FA" >&2
        exit 1
    fi
    if [[ ! -f "$QUERY_FA" ]]; then
        echo "ERROR: DNA query FASTA not found: $QUERY_FA" >&2
        exit 1
    fi
fi
if [[ "$RUN_PROTEIN" -eq 1 ]]; then
    if [[ ! -f "$PROT_TARGET_FA" ]]; then
        echo "ERROR: protein target FASTA not found: $PROT_TARGET_FA" >&2
        exit 1
    fi
    if [[ ! -f "$PROT_QUERY_FA" ]]; then
        echo "ERROR: protein query FASTA not found: $PROT_QUERY_FA" >&2
        exit 1
    fi
fi

run_benchmark_table() {
    local title="$1"
    local seq_type="$2"
    local target_fa="$3"
    local query_fa="$4"
    local modes_str="$5"

    echo
    echo "================================================================"
    echo "  $title"
    echo "================================================================"
    echo "seq_type  : $seq_type"
    echo "target    : $target_fa"
    echo "query     : $query_fa"
    echo "repeats   : $REPEAT"
    echo "band      : $BAND"
    echo "modes     : $modes_str"
    echo "binary    : $PSW_BIN"
    echo

    printf "%-12s | %-14s | %-14s\n" "mode" "avg_time(s)" "avg_mem(MB)"
    printf "%-12s-+-%-14s-+-%-14s\n" "------------" "--------------" "--------------"

    for mode in $modes_str; do
        local sum_time="0"
        local sum_rss_kb=0
        local i elapsed rss_kb avg_time avg_mem_mb

        for ((i=1; i<=REPEAT; i++)); do
            local cmd=("$PSW_BIN" -t "$mode" -w "$BAND")
            if [[ "$seq_type" == "protein" ]]; then
                cmd+=( -S protein )
            fi
            cmd+=("$target_fa" "$query_fa")

            read -r elapsed rss_kb < <({ /usr/bin/time -f "%e %M" \
                "${cmd[@]}" >/dev/null; } 2>&1)

            sum_time="$(awk -v a="$sum_time" -v b="$elapsed" 'BEGIN{printf "%.6f", a+b}')"
            sum_rss_kb=$((sum_rss_kb + rss_kb))
        done

        avg_time="$(awk -v s="$sum_time" -v n="$REPEAT" 'BEGIN{printf "%.6f", s/n}')"
        avg_mem_mb="$(awk -v kb="$sum_rss_kb" -v n="$REPEAT" 'BEGIN{printf "%.3f", (kb/n)/1024.0}')"

        printf "%-12s | %-14s | %-14s\n" "$mode" "$avg_time" "$avg_mem_mb"
    done
}

echo "================================================================"
echo "  Building psw"
echo "================================================================"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_STANDARD=11 -Wno-dev
cmake --build "$BUILD_DIR" --config Release --parallel "$(nproc 2>/dev/null || echo 4)"

if [[ "$RUN_DNA" -eq 1 ]]; then
    run_benchmark_table "Performance benchmark (DNA)" "dna" "$TARGET_FA" "$QUERY_FA" "$MODES_STR"
fi

if [[ "$RUN_PROTEIN" -eq 1 ]]; then
    run_benchmark_table "Performance benchmark (Protein)" "protein" "$PROT_TARGET_FA" "$PROT_QUERY_FA" "$PROT_MODES_STR"
fi

echo
echo "Done."

