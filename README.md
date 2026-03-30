# PSW
## Introduction

`psw` is a sequence alignment project derived from [ksw2](https://github.com/lh3/ksw2), with targeted engineering changes for SIMD-accelerated dynamic programming.

The current focus is practical optimization for DNA workloads, while keeping the codebase extensible for future `profile-profile` and `profile-sequence` alignment work.

## Usage

### 1) Add psw to your build

Compile these files in your project (with responsibilities):
- `psw.h`: public API header; defines core structs (`psw_prof_t`, `psw_extz_t`), flags, CIGAR macros, allocator abstraction (`kmalloc/kfree`), and helper utilities.
- `psw_gg.c`: baseline global profile alignment implementation (`psw_gg_pp`, `psw_gg_ps`).
- `psw_gg2.c`: second global-alignment variant (`psw_gg2_pp`, `psw_gg2_ps`) with a different DP formulation for comparison/tuning.
- `psw_gg3.c`: third scalar global-alignment variant (`psw_gg3_pp`, `psw_gg3_ps`), using scaled integer DP internally.
- `psw_gg3_sse.c`: SSE2-accelerated `gg3` implementation (`psw_gg3_sse_pp`, `psw_gg3_sse_ps`) for x86 SIMD builds.
- `psw_extz.c`: scalar extension/z-drop implementation (`psw_extz_pp`, `psw_extz_ps`) with `psw_extz_t` outputs.
- `psw_extz_sse.c`: SSE2-accelerated extension/z-drop implementation (`psw_extz_sse_pp`, `psw_extz_sse_ps`).

Notes:
- If your project uses `kalloc`, compile with `-DHAVE_KALLOC` and include `kalloc.h`.
- If not, `psw.h` automatically maps allocation to `malloc/calloc/realloc/free`.
- If you do not need SIMD entry points, you can omit `psw_gg3_sse.c` / `psw_extz_sse.c` from your build.

### 2) Minimal integration example

`psw` does not expose a raw sequence-vs-sequence API. The `*_ps` APIs align an encoded query sequence against a target profile (`psw_prof_t`).
So the example below builds a depth-1 target profile from `tseq`, then calls `psw_extz_ps()`.

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "psw.h"

/* Build DNA encoding table: A/C/G/T -> 0/1/2/3, others -> 4 (N) */
static void init_dna_enc(uint8_t enc[256])
{
    memset(enc, 4, 256);
    enc[(uint8_t)'A'] = enc[(uint8_t)'a'] = 0;
    enc[(uint8_t)'C'] = enc[(uint8_t)'c'] = 1;
    enc[(uint8_t)'G'] = enc[(uint8_t)'g'] = 2;
    enc[(uint8_t)'T'] = enc[(uint8_t)'t'] = 3;
}

/* Build a depth-1 target profile from a plain DNA string */
static uint32_t *build_depth1_profile(const char *tseq, const uint8_t enc[256], psw_prof_t *tp)
{
    int i;
    int tl = (int)strlen(tseq);
    const int m = 5; /* A,C,G,T,N */
    uint32_t *prof = (uint32_t*)calloc((size_t)tl * m, sizeof(uint32_t));
    if (!prof) return NULL;

    for (i = 0; i < tl; ++i) {
        int code = enc[(uint8_t)tseq[i]];
        prof[(size_t)i * m + code] = 1;
    }

    tp->len = tl;
    tp->dim = m;
    tp->depth = 1;
    tp->prof = prof;
    return prof;
}

static void print_cigar(const psw_extz_t *ez)
{
    static const char op_table[] = {'M','I','D','N','S','H','?','=','X'};
    int i;
    for (i = 0; i < ez->n_cigar; ++i) {
        int op = (int)(ez->cigar[i] & 0x0fu);
        int len = (int)(ez->cigar[i] >> 4);
        char ch = (op >= 0 && op < (int)(sizeof(op_table) / sizeof(op_table[0]))) ? op_table[op] : '?';
        printf("%d%c", len, ch);
    }
    putchar('\n');
}

int main(void)
{
    const char *tseq = "ATAGCTAGCTAGCAT";
    const char *qseq = "AGCTACCGCAT";
    const int ql = (int)strlen(qseq);
    int i;

    /* match=1, mismatch=-2 (kept negative), gap-open=2, gap-extend=1 */
    const int8_t mat[25] = {
         1,-2,-2,-2,0,
        -2, 1,-2,-2,0,
        -2,-2, 1,-2,0,
        -2,-2,-2, 1,0,
         0, 0, 0, 0,0
    };

    uint8_t enc[256], *qs = NULL;
    uint32_t *tprof_mem = NULL;
    psw_prof_t tp;
    psw_extz_t ez;

    init_dna_enc(enc);

    qs = (uint8_t*)malloc((size_t)ql);
    if (!qs) return 1;
    for (i = 0; i < ql; ++i) qs[i] = enc[(uint8_t)qseq[i]];

    tprof_mem = build_depth1_profile(tseq, enc, &tp);
    if (!tprof_mem) {
        free(qs);
        return 1;
    }

    psw_reset_extz(&ez);
    psw_extz_ps(
        NULL,
        ql, qs,
        tp.len, &tp,
        5, mat,
        2, 1,
        -1, /* disable banded alignment */
        -1, /* disable z-drop */
        PSW_FLAG_GLOBAL,
        &ez
    );

    print_cigar(&ez);

    kfree(NULL, ez.cigar);
    free(tprof_mem);
    free(qs);
    return 0;
}
```

You can switch to SIMD by replacing `psw_extz_ps()` with `psw_extz_sse_ps()`.

### 3) Interface notes (what each argument means)

### `psw_prof_t` in detail

```c
typedef struct {
    int len;
    int dim;
    int depth;              /* number of sequences used to build the profile */
    const uint32_t *prof;   /* per-column counts; each column has dim entries */
} psw_prof_t;
```

`psw_prof_t` is the core container for profile-based alignment in `psw`.

- `len`: number of alignment columns in the profile.
- `dim`: alphabet dimension stored per column.
  - Typical DNA setup in this repo uses `dim=5` for `A,C,G,T,N`.
  - Typical protein setup uses `dim=21` for 20 amino acids + unknown.
- `depth`: how many sequences were used to build this profile.
  - In a depth-1 profile, each non-gap column usually has one count equal to `1`.
  - In multi-sequence profiles, counts represent column-wise residue frequencies.
- `prof`: flattened column-major count matrix.
  - Index rule: `prof[col * dim + sym]`.
  - `col` range: `[0, len-1]`, `sym` range: `[0, dim-1]`.

Interpretation example (`dim=5`, symbol order `A,C,G,T,N`):
- If column `j` has counts `A:3, C:1, G:0, T:0, N:0`, then:
  - `prof[j*5+0]=3`
  - `prof[j*5+1]=1`
  - `prof[j*5+2]=0`
  - `prof[j*5+3]=0`
  - `prof[j*5+4]=0`

Construction requirements for safe API calls:
- `len` must match the real number of profile columns.
- `dim` must be consistent with `m` and `mat` passed to alignment functions (`m <= dim`).
- `prof` must point to at least `len * dim` valid `uint32_t` entries.
- `depth` should be positive; if unknown, use a conservative valid value (for example `1`).

## Test

Run bundled case-based checks:

```bash
bash run_case.sh
```

This script builds `psw` and runs all prepared cases under `test/case/` across multiple modes.

Run performance benchmarks:

```bash
bash run_perf.sh
```

Useful benchmark variants:

```bash
bash run_perf.sh --dna-only
bash run_perf.sh --protein-only
bash run_perf.sh -r 3 -w 128 -m "gg_pp gg3_pp gg3_sse_pp"
```

The default benchmark datasets are under `test/`:
- DNA: `test/MT-human.fa`, `test/MT-orang.fa`
- Protein: `test/protein-perf-target.fa`, `test/protein-perf-query.fa`
