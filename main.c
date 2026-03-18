#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kseq.h"
#include "psw.h"

static int kseq_file_read(FILE *fp, void *buf, int len)
{
    return (int)fread(buf, 1, (size_t)len, fp);
}
KSEQ_INIT(FILE*, kseq_file_read)

#define PSW_DIM 5
#define ALIGN_BLOCK 80

typedef struct {
    char *name;
    char *seq;
    int len;
} fasta_rec_t;

typedef struct {
    const char *mode; /* gg_pp, gg_ps, gg2_pp, gg2_ps, gg3_pp or gg3_ps */
    int8_t match;
    int8_t mismatch;
    int8_t gapo;
    int8_t gape;
    int band;
    int print_alignment;
    const char *target_path;
    const char *query_path;
} cli_opt_t;

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [options] <target.fasta> <query.fasta>\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t STR   mode: gg_pp, gg_ps, gg2_pp, gg2_ps, gg3_pp or gg3_ps [gg_pp]\n");
    fprintf(stderr, "  -A INT   match score [5]\n");
    fprintf(stderr, "  -B INT   mismatch penalty (positive) [4]\n");
    fprintf(stderr, "  -O INT   gap open penalty [6]\n");
    fprintf(stderr, "  -E INT   gap extension penalty [2]\n");
    fprintf(stderr, "  -w INT   band width; -1 disables banding [-1]\n");
    fprintf(stderr, "  -p       print alignment preview (consensus-based)\n");
}

static char *xstrdup(const char *s)
{
    size_t n = strlen(s);
    char *p = (char*)malloc(n + 1);
    if (p == 0) return 0;
    memcpy(p, s, n + 1);
    return p;
}

static int base_to_idx(char c)
{
    switch (c) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    default: return 4; /* N and unknown */
    }
}

static int is_gap_char(char c)
{
    return c == '-' || c == '.';
}

static int idx_to_base(int idx)
{
    static const char table[] = { 'A', 'C', 'G', 'T', 'N' };
    return (idx >= 0 && idx < PSW_DIM) ? (int)table[idx] : (int)'N';
}

static int clamp_i8(int v)
{
    if (v < -128) return -128;
    if (v > 127) return 127;
    return v;
}

static int parse_int_arg(const char *s, int *out)
{
    char *end = 0;
    long v = strtol(s, &end, 10);
    if (s == end || (end && *end != '\0')) return -1;
    *out = (int)v;
    return 0;
}

static void gen_simple_mat(int8_t *mat, int8_t match, int8_t mismatch_penalty)
{
    int i, j;
    int mismatch_i = mismatch_penalty > 0 ? -(int)mismatch_penalty : (int)mismatch_penalty;
    int8_t mismatch = (int8_t)clamp_i8(mismatch_i);

    for (i = 0; i < PSW_DIM - 1; ++i) {
        for (j = 0; j < PSW_DIM - 1; ++j) {
            int8_t v = i == j ? match : mismatch;
            mat[i * PSW_DIM + j] = v;
        }
        mat[i * PSW_DIM + (PSW_DIM - 1)] = 0;
    }
    for (j = 0; j < PSW_DIM; ++j)
        mat[(PSW_DIM - 1) * PSW_DIM + j] = 0;
}

static void free_fasta_records(fasta_rec_t *rec, int n)
{
    int i;
    if (rec == 0) return;
    for (i = 0; i < n; ++i) {
        free(rec[i].name);
        free(rec[i].seq);
    }
    free(rec);
}

static int fasta_push_rec(fasta_rec_t **a, int *n, int *m, const char *name, const char *seq)
{
    fasta_rec_t *tmp;
    if (*n == *m) {
        int new_m = *m ? (*m << 1) : 8;
        tmp = (fasta_rec_t*)realloc(*a, (size_t)new_m * sizeof(fasta_rec_t));
        if (tmp == 0) return -1;
        *a = tmp;
        *m = new_m;
    }
    (*a)[*n].name = xstrdup(name ? name : "seq");
    (*a)[*n].seq = xstrdup(seq ? seq : "");
    if ((*a)[*n].name == 0 || (*a)[*n].seq == 0) {
        free((*a)[*n].name);
        free((*a)[*n].seq);
        return -1;
    }
    (*a)[*n].len = (int)strlen((*a)[*n].seq);
    ++(*n);
    return 0;
}

static int load_fasta(const char *path, fasta_rec_t **out_rec, int *out_n)
{
    FILE *fp = fopen(path, "rb");
    kseq_t *ks = 0;
    fasta_rec_t *rec = 0;
    int n = 0, m = 0;
    int l;

    if (fp == 0) return -1;
    ks = kseq_init(fp);
    if (ks == 0) {
        fclose(fp);
        return -1;
    }

    while ((l = kseq_read(ks)) >= 0) {
        const char *name = ks->name.s ? ks->name.s : "seq";
        const char *seq = ks->seq.s ? ks->seq.s : "";
        if (fasta_push_rec(&rec, &n, &m, name, seq) != 0) {
            free_fasta_records(rec, n);
            kseq_destroy(ks);
            fclose(fp);
            return -1;
        }
        rec[n - 1].len = l;
    }

    kseq_destroy(ks);
    fclose(fp);

    if (l < -1) {
        free_fasta_records(rec, n);
        return -1;
    }

    *out_rec = rec;
    *out_n = n;
    return 0;
}

static int build_profile_from_aligned_fasta(const fasta_rec_t *rec, int n,
                                            uint32_t **out_prof, int *out_len, int *out_depth)
{
    int i, j;
    uint32_t *prof;
    int len;

    if (n <= 0) return -1;
    len = rec[0].len;
    for (i = 1; i < n; ++i)
        if (rec[i].len != len) return -1;

    prof = (uint32_t*)calloc((size_t)len * PSW_DIM, sizeof(uint32_t));
    if (prof == 0) return -1;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < len; ++j) {
            char c = rec[i].seq[j];
            if (is_gap_char(c)) continue; /* implicit gap */
            prof[(size_t)j * PSW_DIM + base_to_idx(c)]++;
        }
    }

    *out_prof = prof;
    *out_len = len;
    *out_depth = n;
    return 0;
}

static int build_query_index_from_fasta(const fasta_rec_t *rec, int n, uint8_t **out_q, int *out_len)
{
    int i, k = 0;
    uint8_t *q;

    if (n != 1) return -1;
    q = (uint8_t*)malloc((size_t)rec[0].len);
    if (q == 0) return -1;

    for (i = 0; i < rec[0].len; ++i) {
        char c = rec[0].seq[i];
        if (is_gap_char(c)) continue;
        q[k++] = (uint8_t)base_to_idx(c);
    }

    *out_q = q;
    *out_len = k;
    return 0;
}

static char *consensus_from_profile(const uint32_t *prof, int len)
{
    int i, b;
    char *s = (char*)malloc((size_t)len + 1);
    if (s == 0) return 0;

    for (i = 0; i < len; ++i) {
        int best = 0;
        uint32_t best_v = 0;
        uint32_t sum = 0;
        for (b = 0; b < PSW_DIM; ++b) {
            uint32_t v = prof[(size_t)i * PSW_DIM + b];
            sum += v;
            if (v > best_v) {
                best_v = v;
                best = b;
            }
        }
        s[i] = (char)(sum == 0 ? '-' : idx_to_base(best));
    }
    s[len] = '\0';
    return s;
}

static void print_cigar(const uint32_t *cigar, int n_cigar)
{
    static const char op_table[] = { 'M', 'I', 'D', 'N', '?', '?', '?', '=', 'X' };
    int i;
    for (i = 0; i < n_cigar; ++i) {
        int op = (int)(cigar[i] & 0xfu);
        unsigned len = cigar[i] >> 4;
        printf("%u%c", len, (op >= 0 && op < (int)(sizeof(op_table) / sizeof(op_table[0]))) ? op_table[op] : '?');
    }
    putchar('\n');
}

static int cigar_alignment_len(const uint32_t *cigar, int n_cigar)
{
    int i, total = 0;
    for (i = 0; i < n_cigar; ++i) {
        int op = (int)(cigar[i] & 0xfu);
        int len = (int)(cigar[i] >> 4);
        if (op == PSW_CIGAR_MATCH || op == PSW_CIGAR_INS || op == PSW_CIGAR_DEL) total += len;
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
        free(aq); free(am); free(at);
        printf("alignment allocation failed\n");
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

    for (i = 0; i < apos; i += ALIGN_BLOCK) {
        int width = apos - i;
        if (width > ALIGN_BLOCK) width = ALIGN_BLOCK;
        printf("Q %.*s\n", width, aq + i);
        printf("  %.*s\n", width, am + i);
        printf("T %.*s\n\n", width, at + i);
    }

    free(aq); free(am); free(at);
}

static int parse_cli(int argc, char **argv, cli_opt_t *opt)
{
    int i, n_pos = 0;
    const char *pos[2] = { 0, 0 };

    for (i = 1; i < argc; ++i) {
        int v;
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) opt->mode = argv[++i];
        else if (strcmp(argv[i], "-A") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &v) != 0) return -1;
            opt->match = (int8_t)clamp_i8(v);
        } else if (strcmp(argv[i], "-B") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &v) != 0) return -1;
            opt->mismatch = (int8_t)clamp_i8(v);
        } else if (strcmp(argv[i], "-O") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &v) != 0) return -1;
            opt->gapo = (int8_t)clamp_i8(v);
        } else if (strcmp(argv[i], "-E") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &v) != 0) return -1;
            opt->gape = (int8_t)clamp_i8(v);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &v) != 0) return -1;
            opt->band = v;
        }
        else if (strcmp(argv[i], "-p") == 0) opt->print_alignment = 1;
        else if (argv[i][0] == '-') return -1;
        else if (n_pos < 2) pos[n_pos++] = argv[i];
        else return -1;
    }

    if (n_pos != 2) return -1;
    opt->target_path = pos[0];
    opt->query_path = pos[1];
    return 0;
}

int main(int argc, char **argv)
{
    cli_opt_t opt;
    int8_t mat[PSW_DIM * PSW_DIM];
    uint32_t *cigar = 0;
    int m_cigar = 0, n_cigar = 0;
    float score = PSW_NEG_INF_F;

    fasta_rec_t *target_rec = 0, *query_rec = 0;
    int n_target = 0, n_query = 0;

    opt.mode = "gg_pp";
    opt.match = 5;
    opt.mismatch = 4;
    opt.gapo = 6;
    opt.gape = 2;
    opt.band = -1;
    opt.print_alignment = 0;
    opt.target_path = 0;
    opt.query_path = 0;

    if (parse_cli(argc, argv, &opt) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(opt.mode, "gg_pp") != 0 && strcmp(opt.mode, "gg_ps") != 0 &&
        strcmp(opt.mode, "gg2_pp") != 0 && strcmp(opt.mode, "gg2_ps") != 0 &&
        strcmp(opt.mode, "gg3_pp") != 0 && strcmp(opt.mode, "gg3_ps") != 0 &&
        strcmp(opt.mode, "pp") != 0 && strcmp(opt.mode, "ps") != 0) {
        fprintf(stderr, "ERROR: -t must be gg_pp, gg_ps, gg2_pp, gg2_ps, gg3_pp or gg3_ps\n");
        return 1;
    }

    if (load_fasta(opt.target_path, &target_rec, &n_target) != 0 || n_target == 0) {
        fprintf(stderr, "ERROR: failed to read target FASTA: %s\n", opt.target_path);
        free_fasta_records(target_rec, n_target);
        return 1;
    }
    if (load_fasta(opt.query_path, &query_rec, &n_query) != 0 || n_query == 0) {
        fprintf(stderr, "ERROR: failed to read query FASTA: %s\n", opt.query_path);
        free_fasta_records(target_rec, n_target);
        free_fasta_records(query_rec, n_query);
        return 1;
    }

    gen_simple_mat(mat, opt.match, opt.mismatch);

    if (strcmp(opt.mode, "gg_pp") == 0 || strcmp(opt.mode, "pp") == 0 ||
        strcmp(opt.mode, "gg2_pp") == 0 || strcmp(opt.mode, "gg3_pp") == 0) {
        uint32_t *target_prof = 0, *query_prof = 0;
        int tlen = 0, qlen = 0;
        int tdepth = 0, qdepth = 0;
        psw_prof_t target, query;
        char *t_cons = 0, *q_cons = 0;

        if (build_profile_from_aligned_fasta(target_rec, n_target, &target_prof, &tlen, &tdepth) != 0 ||
            build_profile_from_aligned_fasta(query_rec, n_query, &query_prof, &qlen, &qdepth) != 0) {
            fprintf(stderr, "ERROR: gg_pp requires aligned FASTA (all records same length within each file)\n");
            free(target_prof); free(query_prof);
            free_fasta_records(target_rec, n_target);
            free_fasta_records(query_rec, n_query);
            return 1;
        }

        target.len = tlen; target.dim = PSW_DIM; target.depth = tdepth; target.prof = target_prof;
        query.len = qlen; query.dim = PSW_DIM; query.depth = qdepth; query.prof = query_prof;

        if (strcmp(opt.mode, "gg2_pp") == 0) {
            score = psw_gg2_pp(0, qlen, &query, tlen, &target, PSW_DIM, mat,
                               opt.gapo, opt.gape, opt.band,
                               &m_cigar, &n_cigar, &cigar);
        } else if (strcmp(opt.mode, "gg3_pp") == 0) {
            score = psw_gg3_pp(0, qlen, &query, tlen, &target, PSW_DIM, mat,
                               opt.gapo, opt.gape, opt.band,
                               &m_cigar, &n_cigar, &cigar);
        } else {
            score = psw_gg_pp(0, qlen, &query, tlen, &target, PSW_DIM, mat,
                              opt.gapo, opt.gape, opt.band,
                              &m_cigar, &n_cigar, &cigar);
        }

        printf("mode=%s score=%.2f\n",
               strcmp(opt.mode, "gg2_pp") == 0 ? "gg2_pp" :
               (strcmp(opt.mode, "gg3_pp") == 0 ? "gg3_pp" : "gg_pp"),
               score);
        printf("target=%s (n_seq=%d, len=%d)\n", opt.target_path, tdepth, tlen);
        printf("query =%s (n_seq=%d, len=%d)\n", opt.query_path, qdepth, qlen);
        printf("cigar: ");
        if (cigar && n_cigar > 0) print_cigar(cigar, n_cigar);
        else printf("<empty>\n");

        if (opt.print_alignment && cigar && n_cigar > 0) {
            q_cons = consensus_from_profile(query_prof, qlen);
            t_cons = consensus_from_profile(target_prof, tlen);
            if (q_cons && t_cons) print_alignment(q_cons, t_cons, cigar, n_cigar);
            free(q_cons); free(t_cons);
        }

        free(target_prof);
        free(query_prof);
    } else {
        uint32_t *target_prof = 0;
        uint8_t *query_idx = 0;
        int tlen = 0, qlen = 0, tdepth = 0;
        psw_prof_t target;
        char *q_str = 0, *t_cons = 0;

        if (build_profile_from_aligned_fasta(target_rec, n_target, &target_prof, &tlen, &tdepth) != 0) {
            fprintf(stderr, "ERROR: gg_ps target must be aligned profile FASTA\n");
            free_fasta_records(target_rec, n_target);
            free_fasta_records(query_rec, n_query);
            return 1;
        }
        if (build_query_index_from_fasta(query_rec, n_query, &query_idx, &qlen) != 0) {
            fprintf(stderr, "ERROR: gg_ps query FASTA must contain exactly one sequence\n");
            free(target_prof);
            free_fasta_records(target_rec, n_target);
            free_fasta_records(query_rec, n_query);
            return 1;
        }

        target.len = tlen; target.dim = PSW_DIM; target.depth = tdepth; target.prof = target_prof;

        if (strcmp(opt.mode, "gg2_ps") == 0) {
            score = psw_gg2_ps(0, qlen, query_idx, tlen, &target, PSW_DIM, mat,
                               opt.gapo, opt.gape, opt.band,
                               &m_cigar, &n_cigar, &cigar);
        } else if (strcmp(opt.mode, "gg3_ps") == 0) {
            score = psw_gg3_ps(0, qlen, query_idx, tlen, &target, PSW_DIM, mat,
                               opt.gapo, opt.gape, opt.band,
                               &m_cigar, &n_cigar, &cigar);
        } else {
            score = psw_gg_ps(0, qlen, query_idx, tlen, &target, PSW_DIM, mat,
                              opt.gapo, opt.gape, opt.band,
                              &m_cigar, &n_cigar, &cigar);
        }

        printf("mode=%s score=%.2f\n",
               strcmp(opt.mode, "gg2_ps") == 0 ? "gg2_ps" :
               (strcmp(opt.mode, "gg3_ps") == 0 ? "gg3_ps" : "gg_ps"),
               score);
        printf("target=%s (n_seq=%d, len=%d)\n", opt.target_path, tdepth, tlen);
        printf("query =%s (n_seq=1, len=%d)\n", opt.query_path, qlen);
        printf("cigar: ");
        if (cigar && n_cigar > 0) print_cigar(cigar, n_cigar);
        else printf("<empty>\n");

        if (opt.print_alignment && cigar && n_cigar > 0) {
            q_str = xstrdup(query_rec[0].seq);
            t_cons = consensus_from_profile(target_prof, tlen);
            if (q_str && t_cons) print_alignment(q_str, t_cons, cigar, n_cigar);
            free(q_str);
            free(t_cons);
        }

        free(query_idx);
        free(target_prof);
    }

    kfree(0, cigar);
    free_fasta_records(target_rec, n_target);
    free_fasta_records(query_rec, n_query);
    return 0;
}
