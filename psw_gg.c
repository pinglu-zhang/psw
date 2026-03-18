//
// Created by 30451 on 2026/3/18.
//#include <stdio.h>   // for debugging only
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
 * 生成每列的 base frequency:
 *   bf[col] = (sum of base counts in this column) / depth
 *
 * 假设 profile 每列前 m 个元素是 A/C/G/T/... 的计数，不含 gap。
 */
static inline float *psw_gen_base_freq(void *km, int len, const psw_prof_t *p, int8_t m)
{
	int i, a;
	float depth;
	float *bf;

	bf = (float*)kmalloc(km, (size_t)len * sizeof(float));
	if (bf == 0) return 0;

	depth = p->depth > 0 ? (float)p->depth : 1.0f;
	for (i = 0; i < len; ++i) {
		const uint32_t *col = p->prof + (size_t)i * p->dim;
		float sum = 0.0f;
		for (a = 0; a < m; ++a)
			sum += (float)col[a];
		bf[i] = sum / depth;
		if (bf[i] < 0.0f) bf[i] = 0.0f;
		if (bf[i] > 1.0f) bf[i] = 1.0f;
	}
	return bf;
}

/*
 * 预计算 query profile
 *
 * 原始列打分:
 *   S(i,j) = sum_a sum_b Q_j[a] * mat[a*m+b] * T_i[b]
 *
 * 预计算后:
 *   qp[b * qlen + j] = (sum_a Q_j[a] * mat[a*m+b]) / query_depth
 *   S(i,j) = (sum_b qp[b * qlen + j] * T_i[b]) / target_depth
 */
static inline float *psw_gen_qp(void *km, int qlen, const psw_prof_t *query, int8_t m, const int8_t *mat)
{
	int a, b, j;
	const float inv_depth = 1.0f / (query->depth > 0 ? (float)query->depth : 1.0f);
	float *qp;

	/* j-major: qp[j * m + b], improves locality when j is inner DP index */
	qp = (float*)kmalloc(km, (size_t)qlen * m * sizeof(float));
	if (qp == 0) return 0;

	for (j = 0; j < qlen; ++j) {
		const uint32_t *qcol = query->prof + (size_t)j * query->dim;
		float *dst = qp + (size_t)j * m;
		for (b = 0; b < m; ++b) {
			float s = 0.0f;
			for (a = 0; a < m; ++a)
				s += (float)qcol[a] * (float)mat[a * m + b];
			dst[b] = s * inv_depth;
		}
	}
	return qp;
}

/*
 * 从预计算的 qp 与 target 第 i 列计算 profile-profile 列打分
 */
static inline float psw_score_from_qp(const float *qp, int j,
                                      const psw_prof_t *target, int i, int8_t m,
                                      float inv_tdepth)
{
	const uint32_t *tcol = target->prof + (size_t)i * target->dim;
	const float *qpj = qp + (size_t)j * m;
	float s = 0.0f;
	int b;

	for (b = 0; b < m; ++b)
		s += qpj[b] * (float)tcol[b];
	return s * inv_tdepth;
}

/*
 * target profile precompute for sequence-profile:
 *   tp[a * tlen + i] = (sum_b mat[a*m+b] * T_i[b]) / target_depth
 */
static inline float *psw_gen_tp(void *km, int tlen, const psw_prof_t *target,
                                int8_t m, const int8_t *mat)
{
	int a, b, i;
	const float inv_depth = 1.0f / (target->depth > 0 ? (float)target->depth : 1.0f);
	float *tp;

	tp = (float*)kmalloc(km, (size_t)m * tlen * sizeof(float));
	if (tp == 0) return 0;

	for (i = 0; i < tlen; ++i) {
		const uint32_t *tcol = target->prof + (size_t)i * target->dim;
		for (a = 0; a < m; ++a) {
			float s = 0.0f;
			for (b = 0; b < m; ++b)
				s += (float)mat[a * m + b] * (float)tcol[b];
			tp[(size_t)a * tlen + i] = s * inv_depth;
		}
	}
	return tp;
}

/*
 * Precompute target profile for pp scoring:
 *   tf[i * m + b] = target_col_i[b] / target_depth
 */
static inline float *psw_gen_tf(void *km, int tlen, const psw_prof_t *target, int8_t m)
{
	int i, b;
	const float inv_depth = 1.0f / (target->depth > 0 ? (float)target->depth : 1.0f);
	float *tf = (float*)kmalloc(km, (size_t)tlen * m * sizeof(float));
	if (tf == 0) return 0;

	for (i = 0; i < tlen; ++i) {
		const uint32_t *tcol = target->prof + (size_t)i * target->dim;
		float *dst = tf + (size_t)i * m;
		for (b = 0; b < m; ++b)
			dst[b] = (float)tcol[b] * inv_depth;
	}
	return tf;
}

/* -------------------------------------------------------------------------- */
/* main                                                                       */
/* -------------------------------------------------------------------------- */

float psw_gg_pp(void *km, int qlen, const psw_prof_t *query,
                int tlen, const psw_prof_t *target,
                int8_t m, const int8_t *mat,
                int8_t gapo, int8_t gape, int w,
                int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
	eh_t *eh;
	float *qp;                  /* query profile */
	float *tf;                  /* target profile as contiguous float rows */
	float *qbf, *tbf;           /* per-column base frequency */
	int32_t i, j, n_col, *off = 0;
	float score;
	uint8_t *z = 0;             /* backtrack matrix; same encoding as ksw2 */
	const float fgapo = (float)gapo, fgape = (float)gape;

	/* rolling boundary H(i+1,0) */
	float ht0;

	/* argument check */
	if (gapo < 0 || gape < 0) return PSW_NEG_INF_F;
	if (query == 0 || target == 0 || mat == 0) return PSW_NEG_INF_F;
	if (query->prof == 0 || target->prof == 0) return PSW_NEG_INF_F;
	if (qlen < 0 || tlen < 0 || m <= 0) return PSW_NEG_INF_F;
	if (query->len < qlen || target->len < tlen) return PSW_NEG_INF_F;
	if (query->dim < m || target->dim < m) return PSW_NEG_INF_F;

	/* allocate memory */
	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = qlen < 2 * w + 1 ? qlen : 2 * w + 1;

	qp = psw_gen_qp(km, qlen, query, m, mat);
	if (qp == 0) return PSW_NEG_INF_F;

	tf = psw_gen_tf(km, tlen, target, m);
	if (tf == 0) {
		kfree(km, qp);
		return PSW_NEG_INF_F;
	}

	qbf = psw_gen_base_freq(km, qlen, query, m);
	if (qbf == 0) {
		kfree(km, qp);
		kfree(km, tf);
		return PSW_NEG_INF_F;
	}

	tbf = psw_gen_base_freq(km, tlen, target, m);
	if (tbf == 0) {
		kfree(km, qp);
		kfree(km, tf);
		kfree(km, qbf);
		return PSW_NEG_INF_F;
	}

	eh = (eh_t*)kcalloc(km, qlen + 1, sizeof(eh_t));
	if (eh == 0) {
		kfree(km, qp);
		kfree(km, tf);
		kfree(km, qbf);
		kfree(km, tbf);
		return PSW_NEG_INF_F;
	}

	if (m_cigar_ && n_cigar_ && cigar_) {
		*n_cigar_ = 0;
		z = (uint8_t*)kmalloc(km, (size_t)n_col * tlen);
		off = (int32_t*)kcalloc(km, tlen, sizeof(int32_t));
		if (z == 0 || off == 0) {
			if (z) kfree(km, z);
			if (off) kfree(km, off);
			kfree(km, qp);
			kfree(km, tf);
			kfree(km, qbf);
			kfree(km, tbf);
			kfree(km, eh);
			return PSW_NEG_INF_F;
		}
	}

	/* fill the first row, keeping the same eh layout as ksw2 */
	{
		float hqj = 0.0f;
		float go_t0 = 0.0f, ge_t0 = 0.0f;

		if (tlen > 0) {
			float st0 = tbf[0];
			go_t0 = fgapo * st0;
			ge_t0 = fgape * st0;
		}

		eh[0].h = 0.0f;
		eh[0].e = tlen > 0 ? -(go_t0 + ge_t0) : PSW_NEG_INF_F;

		for (j = 1; j <= qlen && j <= w; ++j) {
			float sq = qbf[j - 1];
			float go = fgapo * sq;
			float ge = fgape * sq;

			if (j == 1) hqj = -(go + ge);
			else        hqj -= ge;

			eh[j].h = hqj;
			eh[j].e = tlen > 0 ? (hqj - (go_t0 + ge_t0)) : PSW_NEG_INF_F;
		}
		for (; j <= qlen; ++j) {
			eh[j].h = PSW_NEG_INF_F;
			eh[j].e = PSW_NEG_INF_F;
		}
	}

	/* initialize rolling H(0,0) -> H(1,0), H(2,0), ... */
	ht0 = 0.0f;

	/* DP loop */
	for (i = 0; i < tlen; ++i) { /* target in the outer loop */
		float f, h1, go_e, ge_e;
		const float *tfi = tf + (size_t)i * m;
		int32_t st, en;

		/* target-side gap penalties for this row */
		{
			float se = tbf[i];
			go_e = fgapo * se;
			ge_e = fgape * se;
		}

		/* rolling boundary H(i+1,0) */
		if (i == 0) ht0 = -(go_e + ge_e);
		else        ht0 -= ge_e;

		st = i > w ? i - w : 0;
		en = i + w + 1 < qlen ? i + w + 1 : qlen;

		/* row-start boundary: H(i+1,0) and F(i+1,0) */
		h1 = st > 0 ? PSW_NEG_INF_F : ht0;
		if (st > 0 || qlen <= 0) {
			f = PSW_NEG_INF_F;
		} else {
			float sq0 = qbf[0];
			f = ht0 - (fgapo * sq0 + fgape * sq0);
		}

		if (m_cigar_ && n_cigar_ && cigar_) {
			uint8_t *zi = &z[(size_t)i * n_col];
			off[i] = st;

			for (j = st; j < en; ++j) {
				eh_t *p = &eh[j];
				const float *qpj = qp + (size_t)j * m;
				float sq = qbf[j];
				float go_f = fgapo * sq;
				float ge_f = fgape * sq;
				float h = p->h, e = p->e;
				float s = 0.0f;
				uint8_t d;
				int b;

				for (b = 0; b < m; ++b) s += qpj[b] * tfi[b];

				p->h = h1;
				h += s;

				d = h >= e ? 0 : 1;
				h = h >= e ? h : e;

				d = h >= f ? d : 2;
				h = h >= f ? h : f;

				h1 = h; /* H(i,j) */

				{
					float h_open_e = h1 - (go_e + ge_e);
					e -= ge_e;
					d |= e > h_open_e ? 0x08 : 0;
					e  = e > h_open_e ? e : h_open_e;
					p->e = e;
				}

				{
					float h_open_f = h1 - (go_f + ge_f);
					f -= ge_f;
					d |= f > h_open_f ? 0x10 : 0;
					f  = f > h_open_f ? f : h_open_f;
				}

				zi[j - st] = d; /* same encoding as ksw2 */
			}
		} else {
			for (j = st; j < en; ++j) {
				eh_t *p = &eh[j];
				const float *qpj = qp + (size_t)j * m;
				float sq = qbf[j];
				float go_f = fgapo * sq;
				float ge_f = fgape * sq;
				float h = p->h, e = p->e;
				float s = 0.0f;
				int b;

				for (b = 0; b < m; ++b) s += qpj[b] * tfi[b];

				p->h = h1;
				h += s;
				h = h >= e ? h : e;
				h = h >= f ? h : f;
				h1 = h;

				{
					float h_open_e = h1 - (go_e + ge_e);
					e -= ge_e;
					e  = e > h_open_e ? e : h_open_e;
					p->e = e;
				}

				{
					float h_open_f = h1 - (go_f + ge_f);
					f -= ge_f;
					f  = f > h_open_f ? f : h_open_f;
				}
			}
		}

		eh[en].h = h1;
		eh[en].e = PSW_NEG_INF_F;
	}

	score = eh[qlen].h;

	kfree(km, qp);
	kfree(km, tf);
	kfree(km, qbf);
	kfree(km, tbf);
	kfree(km, eh);

	if (m_cigar_ && n_cigar_ && cigar_) {
		psw_backtrack(km, 0, 0, 0, z, off, 0, n_col, tlen - 1, qlen - 1,
		              m_cigar_, n_cigar_, cigar_);
		kfree(km, z);
		kfree(km, off);
	}

	return score;
}

float psw_gg_ps(void *km, int qlen, const uint8_t *query,
				int tlen, const psw_prof_t *target,
				int8_t m, const int8_t *mat,
				int8_t gapo, int8_t gape, int w,
				int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
	eh_t *eh;
	float *tp;                  /* precomputed target score table */
	float *tbf;                 /* target per-column base frequency */
	int32_t i, j, n_col, *off = 0;
	float score;
	uint8_t *z = 0;
	float hq0 = 0.0f;
	const float fgapo = (float)gapo, fgape = (float)gape;

	if (gapo < 0 || gape < 0) return PSW_NEG_INF_F;
	if (query == 0 || target == 0 || mat == 0) return PSW_NEG_INF_F;
	if (target->prof == 0) return PSW_NEG_INF_F;
	if (qlen < 0 || tlen < 0 || m <= 0) return PSW_NEG_INF_F;
	if (target->len < tlen || target->dim < m) return PSW_NEG_INF_F;

	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = qlen < 2 * w + 1 ? qlen : 2 * w + 1;

	tp = psw_gen_tp(km, tlen, target, m, mat);
	if (tp == 0) return PSW_NEG_INF_F;

	tbf = psw_gen_base_freq(km, tlen, target, m);
	if (tbf == 0) {
		kfree(km, tp);
		return PSW_NEG_INF_F;
	}

	/* Validate query alphabet once; avoid checks in the DP hot loop. */
	for (j = 0; j < qlen; ++j) {
		int a = (int)query[j];
		if (a < 0 || a >= m) {
			kfree(km, tp);
			kfree(km, tbf);
			return PSW_NEG_INF_F;
		}
	}

	eh = (eh_t*)kcalloc(km, tlen + 1, sizeof(eh_t));
	if (eh == 0) {
		kfree(km, tp);
		kfree(km, tbf);
		return PSW_NEG_INF_F;
	}

	if (m_cigar_ && n_cigar_ && cigar_) {
		*n_cigar_ = 0;
		z = (uint8_t*)kmalloc(km, (size_t)n_col * tlen);
		off = (int32_t*)kcalloc(km, tlen, sizeof(int32_t));
		if (z == 0 || off == 0) {
			if (z) kfree(km, z);
			if (off) kfree(km, off);
			kfree(km, tp);
			kfree(km, tbf);
			kfree(km, eh);
			return PSW_NEG_INF_F;
		}
		for (i = 0; i < tlen; ++i) off[i] = i > w ? i - w : 0;
	}

	/* initialize first column: H(i,0) and F(i,0) */
	{
		float hi0 = 0.0f;
		eh[0].h = 0.0f;
		eh[0].e = qlen > 0 ? -(fgapo + fgape) : PSW_NEG_INF_F;

		for (i = 1; i <= tlen && i <= w; ++i) {
			float se = tbf[i - 1];
			float go = fgapo * se;
			float ge = fgape * se;

			if (i == 1) hi0 = -(go + ge);
			else        hi0 -= ge;

			eh[i].h = hi0;
			eh[i].e = qlen > 0 ? (hi0 - (fgapo + fgape)) : PSW_NEG_INF_F;
		}
		for (; i <= tlen; ++i) {
			eh[i].h = PSW_NEG_INF_F;
			eh[i].e = PSW_NEG_INF_F;
		}
	}

	for (j = 0; j < qlen; ++j) {
		float f, h1;
		int32_t st, en;
		int a = (int)query[j];
		const float *tpj = tp + (size_t)a * tlen;

		if (j == 0) hq0 = -(fgapo + fgape);
		else        hq0 -= fgape;

		st = j > w ? j - w : 0;
		en = j + w + 1 < tlen ? j + w + 1 : tlen;

		h1 = st > 0 ? PSW_NEG_INF_F : hq0;
		if (st > 0 || tlen <= 0) f = PSW_NEG_INF_F;
		else {
			float se0 = tbf[0];
			f = hq0 - (fgapo * se0 + fgape * se0);
		}

		for (i = st; i < en; ++i) {
			eh_t *p = &eh[i];
			float se = tbf[i];
			float go_e = fgapo * se;
			float ge_e = fgape * se;
			float h = p->h, e = p->e;
			float s = tpj[i];
			uint8_t d;

			p->h = h1;
			h += s;

			d = h >= f ? 0 : 1;
			h = h >= f ? h : f;
			d = h >= e ? d : 2;
			h = h >= e ? h : e;
			h1 = h;

			{
				float h_open_e = h1 - (go_e + ge_e);
				f -= ge_e;
				d |= f > h_open_e ? 0x08 : 0;
				f = f > h_open_e ? f : h_open_e;
			}

			{
				float h_open_f = h1 - (fgapo + fgape);
				e -= fgape;
				d |= e > h_open_f ? 0x10 : 0;
				e = e > h_open_f ? e : h_open_f;
				p->e = e;
			}

			if (z) z[(size_t)i * n_col + (j - off[i])] = d;
		}

		eh[en].h = h1;
		eh[en].e = PSW_NEG_INF_F;
	}

	score = eh[tlen].h;

	kfree(km, tp);
	kfree(km, tbf);
	kfree(km, eh);

	if (z && off) {
		psw_backtrack(km, 0, 0, 0, z, off, 0, n_col, tlen - 1, qlen - 1,
		              m_cigar_, n_cigar_, cigar_);
		kfree(km, z);
		kfree(km, off);
	}

	return score;
}
