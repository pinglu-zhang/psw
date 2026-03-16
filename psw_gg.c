#include <stdio.h>   // for debugging only
#include <stdint.h>
#include "psw.h"

typedef struct { float h, e; } eh_t;

/*
 * psw.h 里直接复用了 ksw_backtrack()/ksw_push_cigar()，
 * 这些 helper 仍然使用 KSW_CIGAR_* 宏。
 * 这里做兼容映射，避免单独改头文件也能编译。
 */
#ifndef KSW_CIGAR_MATCH
#define KSW_CIGAR_MATCH PSW_CIGAR_MATCH
#endif
#ifndef KSW_CIGAR_INS
#define KSW_CIGAR_INS   PSW_CIGAR_INS
#endif
#ifndef KSW_CIGAR_DEL
#define KSW_CIGAR_DEL   PSW_CIGAR_DEL
#endif
#ifndef KSW_CIGAR_EQ
#define KSW_CIGAR_EQ    PSW_CIGAR_EQ
#endif
#ifndef KSW_CIGAR_X
#define KSW_CIGAR_X     PSW_CIGAR_X
#endif
#ifndef KSW_CIGAR_N_SKIP
#define KSW_CIGAR_N_SKIP 3
#endif

#ifndef PSW_NEG_INF_F
#define PSW_NEG_INF_F (-1e30f)
#endif

/*
 * 预计算 query profile
 *
 * 原始列打分:
 *   S(i,j) = sum_a sum_b Q_j[a] * mat[a*m+b] * T_i[b]
 *
 * 预计算后:
 *   qp[b * qlen + j] = sum_a Q_j[a] * mat[a*m+b]
 *   S(i,j) = sum_b qp[b * qlen + j] * T_i[b]
 */
static inline float *psw_gen_qp(void *km, int qlen, const psw_prof_t *query, int8_t m, const int8_t *mat)
{
	int a, b, j;
	float *qp;
	qp = (float*)kmalloc(km, (size_t)m * qlen * sizeof(float));
	if (qp == 0) return 0;
	for (j = 0; j < qlen; ++j) {

		const uint32_t *qcol = query->prof + (size_t)j * query->dim;
		/* 计算列总和 */
		float sum = 0.0f;
		for (a = 0; a < m; ++a)
			sum += (float)qcol[a];
		if (sum == 0.0f) sum = 1.0f;

		for (b = 0; b < m; ++b) {
			float s = 0.0f;
			for (a = 0; a < m; ++a) {
				float freq = (float)qcol[a] / sum;
				s += freq * (float)mat[a * m + b];
			}
			qp[(size_t)b * qlen + j] = s;
		}
	}
	return qp;
}

/*
 * 从预计算的 qp 与 target 第 i 列计算 profile-profile 列打分
 */
static inline float psw_score_from_qp(const float *qp, int qlen, int j,
                                      const psw_prof_t *target, int i, int8_t m)
{
	const uint32_t *tcol = target->prof + (size_t)i * target->dim;
	float s = 0.0f;
	int b;

	for (b = 0; b < m; ++b)
		s += qp[(size_t)b * qlen + j] * (float)tcol[b];

	return s;
}

float psw_gg_pp(void *km, int qlen, const psw_prof_t *query, int tlen, const psw_prof_t *target,
                int8_t m, const int8_t *mat, int8_t gapo, int8_t gape, int w,
                int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
	eh_t *eh = 0;
	float *qp = 0;
	int32_t i, j, n_col, *off = 0;
	float gapoe, score;
	uint8_t *z = 0; // backtrack matrix; each cell: f<<4 | e<<3 | h(low 2 bits)

	/* argument check */
	if (query == 0 || target == 0 || mat == 0) return PSW_NEG_INF_F;
	if (query->prof == 0 || target->prof == 0) return PSW_NEG_INF_F;
	if (qlen < 0 || tlen < 0 || m <= 0) return PSW_NEG_INF_F;
	if (query->len < qlen || target->len < tlen) return PSW_NEG_INF_F;
	if (query->dim < m || target->dim < m) return PSW_NEG_INF_F;

	gapoe = (float)gapo + (float)gape;

	/* allocate memory */
	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = qlen < 2 * w + 1 ? qlen : 2 * w + 1;

	qp = psw_gen_qp(km, qlen, query, m, mat);
	if (qp == 0) return PSW_NEG_INF_F;

	eh = (eh_t*)kcalloc(km, qlen + 1, sizeof(eh_t));
	if (eh == 0) {
		kfree(km, qp);
		return PSW_NEG_INF_F;
	}

	if (m_cigar_ && n_cigar_ && cigar_) {
		*n_cigar_ = 0;
		z = (uint8_t*)kmalloc(km, (size_t)n_col * tlen);
		off = (int32_t*)kcalloc(km, tlen, sizeof(int32_t));
		if (z == 0 || off == 0) {
			if (z) kfree(km, z);
			if (off) kfree(km, off);
			kfree(km, eh);
			kfree(km, qp);
			return PSW_NEG_INF_F;
		}
	}

	/* fill the first row */
	eh[0].h = 0.0f;
	eh[0].e = -gapoe - gapoe;
	for (j = 1; j <= qlen && j <= w; ++j) {
		eh[j].h = -(gapoe + (float)gape * (float)(j - 1));
		eh[j].e = -(gapoe + gapoe + (float)gape * (float)j);
	}
	for (; j <= qlen; ++j) {
		eh[j].h = PSW_NEG_INF_F;
		eh[j].e = PSW_NEG_INF_F;
	}

	/* DP loop */
	for (i = 0; i < tlen; ++i) { /* target in outer loop */
		float f, h1;
		int32_t st, en;

		st = i > w ? i - w : 0;
		en = i + w + 1 < qlen ? i + w + 1 : qlen;

		h1 = st > 0 ? PSW_NEG_INF_F : -(gapoe + (float)gape * (float)i);
		f  = st > 0 ? PSW_NEG_INF_F : -(gapoe + gapoe + (float)gape * (float)i);

		if (m_cigar_ && n_cigar_ && cigar_) {
			uint8_t *zi = &z[(size_t)i * n_col];
			off[i] = st;

			for (j = st; j < en; ++j) {
				/*
				 * loop start:
				 *   eh[j].h = H(i-1, j-1)
				 *   eh[j].e = E(i,   j)
				 *   f       = F(i,   j)
				 *   h1      = H(i,   j-1)
				 */
				eh_t *p = &eh[j];
				float h = p->h, e = p->e;
				float s = psw_score_from_qp(qp, qlen, j, target, i, m);
				uint8_t d;

				p->h = h1;
				h += s;

				d = h >= e ? 0 : 1;
				h = h >= e ? h : e;

				d = h >= f ? d : 2;
				h = h >= f ? h : f;

				h1 = h; /* H(i,j) */

				h -= gapoe;
				e -= (float)gape;
				d |= e > h ? 0x08 : 0;
				e  = e > h ? e : h;
				p->e = e; /* E(i+1,j) */

				f -= (float)gape;
				d |= f > h ? 0x10 : 0;
				f  = f > h ? f : h; /* F(i,j+1) */

				zi[j - st] = d;
			}
		} else {
			for (j = st; j < en; ++j) {
				eh_t *p = &eh[j];
				float h = p->h, e = p->e;
				float s = psw_score_from_qp(qp, qlen, j, target, i, m);

				p->h = h1;
				h += s;
				h = h >= e ? h : e;
				h = h >= f ? h : f;
				h1 = h;

				h -= gapoe;
				e -= (float)gape;
				e  = e > h ? e : h;
				p->e = e;

				f -= (float)gape;
				f  = f > h ? f : h;
			}
		}

		eh[en].h = h1;
		eh[en].e = PSW_NEG_INF_F;
	}

	score = eh[qlen].h;

	kfree(km, qp);
	kfree(km, eh);

	if (m_cigar_ && n_cigar_ && cigar_) {
		ksw_backtrack(km, 0, 0, 0, z, off, 0, n_col, tlen - 1, qlen - 1,
		              m_cigar_, n_cigar_, cigar_);
		kfree(km, z);
		kfree(km, off);
	}

	return score;
}