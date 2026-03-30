#!/usr/bin/env bash
# run_case.sh — Build psw then run all test cases under test/case/

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
    local mode="$1"     # gg_pp/gg_ps/gg2_pp/gg2_ps/gg3_pp/gg3_ps/gg3_sse_pp/gg3_sse_ps/sw_pp/sw_ps/extz_pp/extz_ps/extz_sse_pp/extz_sse_ps
    local name="$2"     # case folder name under gg_pp/ or gg_ps/
    local target="$3"
    local query="$4"
    shift 4

    echo "----------------------------------------------------------------"
    echo "  [$mode] $name"
    echo "  target : $target"
    echo "  query  : $query"
    echo "----------------------------------------------------------------"
    "$PSW" -t "$mode" "$@" -p "$target" "$query" || true
    echo ""
}

# Identify protein test case by folder name and pass sequence type explicitly.
case_extra_args() {
    local name="$1"
    if [[ "$name" == "protein_case" ]]; then
        echo "-S protein"
    fi
}

pick_query_file_ps() {
    local case_dir="$1"
    if [[ -f "$case_dir/query_seq.fa" ]]; then
        echo "$case_dir/query_seq.fa"
    else
        echo "$case_dir/query_aln.fa"
    fi
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
    extra_args=( $(case_extra_args "$name") )
    run_case "gg_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa" \
        "${extra_args[@]}"
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
    query_file="$(pick_query_file_ps "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "gg_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$query_file" \
        "${extra_args[@]}"
done

echo "================================================================"
echo "  gg2_pp cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_pp"/*/; do
    name="$(basename "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "gg2_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa" \
        "${extra_args[@]}"
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
    query_file="$(pick_query_file_ps "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "gg2_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$query_file" \
        "${extra_args[@]}"
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
    extra_args=( $(case_extra_args "$name") )
    run_case "gg3_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa" \
        "${extra_args[@]}"
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
    query_file="$(pick_query_file_ps "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "gg3_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$query_file" \
        "${extra_args[@]}"
done

echo "================================================================"
echo "  All cases finished"
echo "================================================================"


echo "================================================================"
echo "  gg3_sse_pp cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_pp"/*/; do
    name="$(basename "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "gg3_sse_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa" \
        "${extra_args[@]}"
done

# --------------------------------------------------------------------------
# 5. sw_pp cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  sw_pp cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_pp"/*/; do
    name="$(basename "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "sw_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa" \
        "${extra_args[@]}"
done

# --------------------------------------------------------------------------
# 4. sw_ps cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  sw_ps cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_ps"/*/; do
    name="$(basename "$case_dir")"
    query_file="$(pick_query_file_ps "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "sw_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$query_file" \
        "${extra_args[@]}"
done

# --------------------------------------------------------------------------
# 5. extz_pp cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  extz_pp cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_pp"/*/; do
    name="$(basename "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "extz_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa" \
        "${extra_args[@]}"
done

# --------------------------------------------------------------------------
# 6. extz_sse_pp cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  extz_sse_pp cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_pp"/*/; do
    name="$(basename "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "extz_sse_pp" "$name" \
        "$case_dir/target_aln.fa" \
        "$case_dir/query_aln.fa" \
        "${extra_args[@]}"
done

# --------------------------------------------------------------------------
# 7. gg3_sse_ps cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  gg3_sse_ps cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_ps"/*/; do
    name="$(basename "$case_dir")"
    query_file="$(pick_query_file_ps "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "gg3_sse_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$query_file" \
        "${extra_args[@]}"
done

# --------------------------------------------------------------------------
# 8. extz_ps cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  extz_ps cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_ps"/*/; do
    name="$(basename "$case_dir")"
    query_file="$(pick_query_file_ps "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "extz_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$query_file" \
        "${extra_args[@]}"
done

# --------------------------------------------------------------------------
# 9. extz_sse_ps cases
# --------------------------------------------------------------------------
echo "================================================================"
echo "  extz_sse_ps cases"
echo "================================================================"
echo ""

for case_dir in "$CASE_DIR/gg_ps"/*/; do
    name="$(basename "$case_dir")"
    query_file="$(pick_query_file_ps "$case_dir")"
    extra_args=( $(case_extra_args "$name") )
    run_case "extz_sse_ps" "$name" \
        "$case_dir/target_aln.fa" \
        "$query_file" \
        "${extra_args[@]}"
done

echo "================================================================"
echo "  All cases finished"
echo "================================================================"

