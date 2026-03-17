#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "psw.h"

#define DEMO_DIM 5
#define ALIGN_BLOCK 80

static int base_to_idx(char c)
{
    switch (c) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    case 'N': case 'n': return 4;
    default: return 4;
    }
}

static void sequence_to_profile(const char *seq, int len, uint32_t *prof,
                                int depth, int minor_pct, int gap_pct)
{
    int i, d;
    if (depth < 1) depth = 1;
    if (minor_pct < 0) minor_pct = 0;
    if (minor_pct > 100) minor_pct = 100;
    if (gap_pct < 0) gap_pct = 0;
    if (gap_pct > 100) gap_pct = 100;

    for (i = 0; i < len; ++i) {
        int major = base_to_idx(seq[i]);
        int effective_depth = depth * (100 - gap_pct) / 100;
        int minor_total = effective_depth * minor_pct / 100;
        int major_count = effective_depth - minor_total;
        int each_minor = minor_total / (DEMO_DIM - 1);
        int rem_minor = minor_total % (DEMO_DIM - 1);
        uint32_t *col = prof + (size_t)i * DEMO_DIM;

        for (d = 0; d < DEMO_DIM; ++d) col[d] = 0;
        if (effective_depth <= 0) continue; /* this column is all implicit gap */
        col[major] = (uint32_t)major_count;

        for (d = 1; d < DEMO_DIM; ++d) {
            int alt = (major + d) % DEMO_DIM;
            int add = each_minor + (d <= rem_minor ? 1 : 0);
            col[alt] += (uint32_t)add;
        }
    }
}

static void print_cigar(const uint32_t *cigar, int n_cigar)
{
    static const char op_table[] = { 'M', 'I', 'D', 'N', '?', '?', '?', '=', 'X' };
    int i;
    for (i = 0; i < n_cigar; ++i) {
        int op = (int)(cigar[i] & 0xfu);
        unsigned len = cigar[i] >> 4;
        const char *op_str = (op >= 0 && op < (int)(sizeof(op_table) / sizeof(op_table[0]))) ? &op_table[op] : "?";
        printf("%u%c", len, *op_str);
    }
    printf("\n");
}

static int cigar_alignment_len(const uint32_t *cigar, int n_cigar)
{
    int i, total = 0;
    for (i = 0; i < n_cigar; ++i) {
        int op = (int)(cigar[i] & 0xfu);
        unsigned len = cigar[i] >> 4;
        if (op == PSW_CIGAR_MATCH || op == PSW_CIGAR_INS || op == PSW_CIGAR_DEL) total += (int)len;
    }
    return total;
}

static void print_alignment(const char *query, const char *target, const uint32_t *cigar, int n_cigar)
{
    int i, k, qpos = 0, tpos = 0, apos = 0;
    int alen = cigar_alignment_len(cigar, n_cigar);
    char *aq = (char*)malloc((size_t)alen + 1);
    char *am = (char*)malloc((size_t)alen + 1);
    char *at = (char*)malloc((size_t)alen + 1);

    if (!aq || !am || !at) {
        printf("alignment memory allocation failed\n");
        free(aq);
        free(am);
        free(at);
        return;
    }

    for (i = 0; i < n_cigar; ++i) {
        int op = (int)(cigar[i] & 0xfu);
        int len = (int)(cigar[i] >> 4);
        for (k = 0; k < len; ++k) {
            if (op == PSW_CIGAR_MATCH) {
                aq[apos] = query[qpos++];
                at[apos] = target[tpos++];
                am[apos] = (aq[apos] == at[apos]) ? '|' : '.';
                ++apos;
            } else if (op == PSW_CIGAR_INS) {
                aq[apos] = query[qpos++];
                at[apos] = '-';
                am[apos] = ' ';
                ++apos;
            } else if (op == PSW_CIGAR_DEL) {
                aq[apos] = '-';
                at[apos] = target[tpos++];
                am[apos] = ' ';
                ++apos;
            }
        }
    }

    aq[apos] = '\0';
    am[apos] = '\0';
    at[apos] = '\0';

    printf("alignment:\n");
    for (i = 0; i < apos; i += ALIGN_BLOCK) {
        int width = apos - i;
        if (width > ALIGN_BLOCK) width = ALIGN_BLOCK;
        printf("Q %.*s\n", width, aq + i);
        printf("  %.*s\n", width, am + i);
        printf("T %.*s\n\n", width, at + i);
    }

    free(aq);
    free(am);
    free(at);
}

static void run_case(const char *name, const char *query_seq, const char *target_seq,
                     int depth, int minor_pct, int gap_pct)
{
    static const int8_t mat[DEMO_DIM * DEMO_DIM] = {
         2, -2, -2, -2,  0,
        -2,  2, -2, -2,  0,
        -2, -2,  2, -2,  0,
        -2, -2, -2,  2,  0,
         0,  0,  0,  0,  0
    };

    int qlen = (int)strlen(query_seq);
    int tlen = (int)strlen(target_seq);
    uint32_t *query_prof = (uint32_t*)calloc((size_t)qlen * DEMO_DIM, sizeof(uint32_t));
    uint32_t *target_prof = (uint32_t*)calloc((size_t)tlen * DEMO_DIM, sizeof(uint32_t));
    psw_prof_t query = { qlen, DEMO_DIM, depth, query_prof };
    psw_prof_t target = { tlen, DEMO_DIM, depth, target_prof };
    int m_cigar = 0, n_cigar = 0;
    uint32_t *cigar = 0;
    float score;

    if (!query_prof || !target_prof) {
        printf("profile allocation failed\n");
        free(query_prof);
        free(target_prof);
        return;
    }

    sequence_to_profile(query_seq, qlen, query_prof, depth, minor_pct, gap_pct);
    sequence_to_profile(target_seq, tlen, target_prof, depth, minor_pct, gap_pct);

    score = psw_gg_pp(0, query.len, &query, target.len, &target,
                      DEMO_DIM, mat, 4, 2, -1,
                      &m_cigar, &n_cigar, &cigar);

    printf("=== %s ===\n", name);
    printf("query (%d):  %s\n", qlen, query_seq);
    printf("target(%d):  %s\n", tlen, target_seq);
    printf("score: %.2f\n", score);
    printf("profile: depth=%d minor=%d%% gap=%d%%(implicit)\n", depth, minor_pct, gap_pct);
    printf("cigar: ");
    if (cigar && n_cigar > 0) print_cigar(cigar, n_cigar);
    else printf("<empty>\n");
    if (cigar && n_cigar > 0) print_alignment(query_seq, target_seq, cigar, n_cigar);

    kfree(0, cigar);
    free(query_prof);
    free(target_prof);
}

int main(void)
{
    run_case("Case 1: exact match with N",
             "ACGTTGACCTGAACTGACGNTACGATGCTA",
             "ACGTTGACCTGAACTGACGNTACGATGCTA",
             100, 20, 15);

    run_case("Case 2: SNP + indel + N",
             "ACGTTGACCTGAACTGACGNTACGATGCTA",
             "ACGTCGACCTGAATGACNCTACGATGCTA",
             100, 20, 15);

    run_case("Case 3: gap-dominant profile",
             "NNNNACGTNNNNACGTNNNN",
             "NNNNACGNNNNNACGTNNNN",
             100, 10, 85);

    run_case("Case 4: high ambiguity (N-rich)",
             "ACNNNTGACNNNTTGANNNAC",
             "ACNNTTGACNNNCTGANNNAC",
             120, 35, 30);

    run_case("Case 5: severe indel stress",
             "ACGTACGTACGTACGTACGTACGT",
             "ACGTACGTTTACGTACGT",
             90, 15, 40);
    return 0;
}
