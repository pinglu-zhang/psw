#ifndef PSW_H_
#define PSW_H_

#include <stdint.h>

#define PSW_NEG_INF -0x40000000

#define PSW_FLAG_SCORE_ONLY   0x01
#define PSW_FLAG_GLOBAL       0x02
#define PSW_FLAG_SEMIGLOBAL   0x04
#define PSW_FLAG_LOCAL        0x08
#define PSW_FLAG_REV_CIGAR    0x10
#define PSW_FLAG_EQX          0x20

#define PSW_CIGAR_MATCH  0
#define PSW_CIGAR_INS    1
#define PSW_CIGAR_DEL    2
#define PSW_CIGAR_EQ     7
#define PSW_CIGAR_X      8

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        uint32_t max:31, zdropped:1;
        int max_q, max_t;
        int score;
        int m_cigar, n_cigar;
        uint32_t *cigar;
    } psw_extz_t;

    typedef struct {
        int len;
        int dim;
        const int32_t *prof;   // dim * len
    } psw_prof_t;


    /**
     * NW-like extension
     *
     * @param km        memory pool, when used with kalloc
     * @param qlen      query length
     * @param query      query profile; query->prof stores q_len columns, each column has m floats
     * @param tlen      target length
     * @param target    target profile; target->prof stores t_len columns, each column has m floats
     * @param m         number of residue types
     * @param mat       m*m scoring mattrix in one-dimension array
     * @param gapo      gap open penalty; a gap of length l cost "-(gapo+l*gape)"
     * @param gape      gap extension penalty
     * @param w         band width (<0 to disable)
     * @param zdrop     off-diagonal drop-off to stop extension (positive; <0 to disable)
     * @param flag      flag (see KSW_EZ_* macros)
     * @param ez        (out) scores and cigar
     */
    void psw_extz(void *km, int q_len, const psw_prof_t *query, int t_len, const psw_prof_t *target, int8_t m, const int8_t *mat,
              int8_t q, int8_t e, int w, int zdrop, int flag, psw_extz_t *ez);

    /**
    * Global alignment
    *
    * (first 10 parameters identical to ksw_extz_sse())
    * @param m_cigar   (modified) max CIGAR length; feed 0 if cigar==0
    * @param n_cigar   (out) number of CIGAR elements
    * @param cigar     (out) BAM-encoded CIGAR; caller need to deallocate with kfree(km, )
    *
    * @return          score of the alignment
    */
	float psw_gg_pp(void *km, int qlen, const psw_prof_f *query, int tlen, const psw_prof_f *target, int8_t m, const float *mat,
		float gapo, float gape, int w, int *m_cigar_, int *n_cigar_, uint32_t **cigar_);

#ifdef __cplusplus
}
#endif
#endif

#ifdef HAVE_KALLOC
#include "kalloc.h"
#else
#include <stdlib.h>
#define kmalloc(km, size) malloc((size))
#define kcalloc(km, count, size) calloc((count), (size))
#define krealloc(km, ptr, size) realloc((ptr), (size))
#define kfree(km, ptr) free((ptr))
#endif



static inline uint32_t *ksw_push_cigar(void *km, int *n_cigar, int *m_cigar, uint32_t *cigar, uint32_t op, int len)
{
	if (*n_cigar == 0 || op != (cigar[(*n_cigar) - 1]&0xf)) {
		if (*n_cigar == *m_cigar) {
			*m_cigar = *m_cigar? (*m_cigar)<<1 : 4;
			cigar = (uint32_t*)krealloc(km, cigar, (*m_cigar) << 2);
		}
		cigar[(*n_cigar)++] = len<<4 | op;
	} else cigar[(*n_cigar)-1] += len<<4;
	return cigar;
}

// In the backtrack matrix, value p[] has the following structure:
//   bit 0-2: which type gets the max - 0 for H, 1 for E, 2 for F, 3 for \tilde{E} and 4 for \tilde{F}
//   bit 3/0x08: 1 if a continuation on the E state (bit 5/0x20 for a continuation on \tilde{E})
//   bit 4/0x10: 1 if a continuation on the F state (bit 6/0x40 for a continuation on \tilde{F})
static inline void ksw_backtrack(void *km, int is_rot, int is_rev, int min_intron_len, const uint8_t *p, const int *off, const int *off_end, int n_col, int i0, int j0,
								 int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{ // p[] - lower 3 bits: which type gets the max; bit
	int n_cigar = 0, m_cigar = *m_cigar_, i = i0, j = j0, r, state = 0;
	uint32_t *cigar = *cigar_, tmp;
	while (i >= 0 && j >= 0) { // at the beginning of the loop, _state_ tells us which state to check
		int force_state = -1;
		if (is_rot) {
			r = i + j;
			if (i < off[r]) force_state = 2;
			if (off_end && i > off_end[r]) force_state = 1;
			tmp = force_state < 0? p[(size_t)r * n_col + i - off[r]] : 0;
		} else {
			if (j < off[i]) force_state = 2;
			if (off_end && j > off_end[i]) force_state = 1;
			tmp = force_state < 0? p[(size_t)i * n_col + j - off[i]] : 0;
		}
		if (state == 0) state = tmp & 7; // if requesting the H state, find state one maximizes it.
		else if (!(tmp >> (state + 2) & 1)) state = 0; // if requesting other states, _state_ stays the same if it is a continuation; otherwise, set to H
		if (state == 0) state = tmp & 7; // TODO: probably this line can be merged into the "else if" line right above; not 100% sure
		if (force_state >= 0) state = force_state;
		if (state == 0) cigar = ksw_push_cigar(km, &n_cigar, &m_cigar, cigar, KSW_CIGAR_MATCH, 1), --i, --j;
		else if (state == 1 || (state == 3 && min_intron_len <= 0)) cigar = ksw_push_cigar(km, &n_cigar, &m_cigar, cigar, KSW_CIGAR_DEL, 1), --i;
		else if (state == 3 && min_intron_len > 0) cigar = ksw_push_cigar(km, &n_cigar, &m_cigar, cigar, KSW_CIGAR_N_SKIP, 1), --i;
		else cigar = ksw_push_cigar(km, &n_cigar, &m_cigar, cigar, KSW_CIGAR_INS, 1), --j;
	}
	if (i >= 0) cigar = ksw_push_cigar(km, &n_cigar, &m_cigar, cigar, min_intron_len > 0 && i >= min_intron_len? KSW_CIGAR_N_SKIP : KSW_CIGAR_DEL, i + 1); // first deletion
	if (j >= 0) cigar = ksw_push_cigar(km, &n_cigar, &m_cigar, cigar, KSW_CIGAR_INS, j + 1); // first insertion
	if (!is_rev)
		for (i = 0; i < n_cigar>>1; ++i) // reverse CIGAR
			tmp = cigar[i], cigar[i] = cigar[n_cigar-1-i], cigar[n_cigar-1-i] = tmp;
	*m_cigar_ = m_cigar, *n_cigar_ = n_cigar, *cigar_ = cigar;
}

static inline void ksw_cigar2eqx(void *km, const uint8_t *query, const uint8_t *target, int nc0, const uint32_t *ci0, int *mc1, int *nc1, uint32_t **ci1)
{
	int i, k, x = 0, y = 0;
	*nc1 = 0;
	for (k = 0; k < nc0; ++k) {
		int op = ci0[k]&0xf, len = ci0[k]>>4;
		if (op == KSW_CIGAR_MATCH) {
			for (i = 0; i < len; ++i) {
				if (target[x + i] == query[y + i]) ksw_push_cigar(km, nc1, mc1, *ci1, KSW_CIGAR_EQ, 1);
				else ksw_push_cigar(km, nc1, mc1, *ci1, KSW_CIGAR_X, 1);
			}
			x += len, y += len;
		} else {
			ksw_push_cigar(km, nc1, mc1, *ci1, op, len);
			if (op == KSW_CIGAR_DEL || op == KSW_CIGAR_N_SKIP) x += len;
			else if (op == KSW_CIGAR_INS) y += len;
			else if (op == KSW_CIGAR_EQ || op == KSW_CIGAR_X) x += len, y += len;
		}
	}
}

static inline void ksw_reset_extz(ksw_extz_t *ez)
{
	ez->max_q = ez->max_t = ez->mqe_t = ez->mte_q = -1;
	ez->max = 0, ez->score = ez->mqe = ez->mte = KSW_NEG_INF;
	ez->n_cigar = 0, ez->zdropped = 0, ez->reach_end = 0;
}

static inline int ksw_apply_zdrop(ksw_extz_t *ez, int is_rot, int32_t H, int a, int b, int zdrop, int8_t e)
{
	int r, t;
	if (is_rot) r = a, t = b;
	else r = a + b, t = a;
	if (H > (int32_t)ez->max) {
		ez->max = H, ez->max_t = t, ez->max_q = r - t;
	} else if (t >= ez->max_t && r - t >= ez->max_q) {
		int tl = t - ez->max_t, ql = (r - t) - ez->max_q, l;
		l = tl > ql? tl - ql : ql - tl;
		if (zdrop >= 0 && ez->max - H > zdrop + l * e) {
			ez->zdropped = 1;
			return 1;
		}
	}
	return 0;
}
#endif