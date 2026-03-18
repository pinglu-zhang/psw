#!/usr/bin/env bash
# run_case.sh — Build psw then run all test cases under test/case/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_ROOT/build"
CASE_DIR="$PROJECT_ROOT/test/case"
PSW="$BUILD_DIR/psw"

# --------------------------------------------------------------------------
# 1. Build
# --------------------------------------------------------------------------
echo "================================================================"
echo "  Building psw"
echo "================================================================"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_STANDARD=11 -Wno-dev
cmake --build "$BUILD_DIR" --config Release --parallel "$(nproc 2>/dev/null || echo 4)"
echo "Binary: $PSW"
echo ""

# --------------------------------------------------------------------------
# 2. Helper: run one case and print a banner
# --------------------------------------------------------------------------
run_case() {
    local mode="$1"     # gg_pp or gg_ps
    local name="$2"     # case folder name under gg_pp/ or gg_ps/
    local target="$3"
    local query="$4"

    echo "----------------------------------------------------------------"
    echo "  [$mode] $name"
    echo "  target : $target"
    echo "  query  : $query"
    echo "----------------------------------------------------------------"
    "$PSW" -t "$mode" -p "$target" "$query" || true
    echo ""
}

# --------------------------------------------------------------------------
# 3. gg_pp cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  gg_pp cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_pp"/*/; do
    name="$(basename "$case_dir")"
    run_case "gg_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa"
done

# --------------------------------------------------------------------------
# 4. gg_ps cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  gg_ps cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_ps"/*/; do
    name="$(basename "$case_dir")"
    run_case "gg_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_seq.fa"
done

echo "================================================================"
echo "  gg2_pp cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_pp"/*/; do
    name="$(basename "$case_dir")"
    run_case "gg2_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa"
done

# --------------------------------------------------------------------------
# 4. gg_ps cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  gg2_ps cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_ps"/*/; do
    name="$(basename "$case_dir")"
    run_case "gg2_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_seq.fa"
done

echo "================================================================"
echo "  All cases finished"
echo "================================================================"


echo "================================================================"
echo "  gg3_pp cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_pp"/*/; do
    name="$(basename "$case_dir")"
    run_case "gg3_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa"
done

# --------------------------------------------------------------------------
# 4. gg_ps cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  gg3_ps cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_ps"/*/; do
    name="$(basename "$case_dir")"
    run_case "gg3_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_seq.fa"
done

echo "================================================================"
echo "  All cases finished"
echo "================================================================"


echo "================================================================"
echo "  All cases finished"
echo "================================================================"

